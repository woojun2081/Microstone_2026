import os
import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pipeline import run_evaluation_pipeline

st.set_page_config(page_title="AI 지능형 보험 추천 시스템", layout="wide", page_icon="🛡️")

# -------------------------------------------------------------
# 1. 공공데이터 로드 및 기준선 산출
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
stat_path = os.path.join(BASE_DIR, "데이터 2차 정제 결과.csv")
products_path = os.path.join(BASE_DIR, "products.json")

@st.cache_data
def load_public_data():
    try:
        df = pd.read_csv(stat_path, encoding="utf-8")
    except:
        df = pd.read_csv(stat_path, encoding="cp949")
    
    df["총진료부담액"] = df["환자수"] * df["1인당_평균진료금"]
    return df

df_stat = load_public_data()

def get_age_group(age: int) -> str:
    if age < 5: return "1_4세"
    elif age >= 80: return "80세 이상"
    else:
        lower = (age // 5) * 5
        return f"{lower}_{lower + 4}세"

def get_public_baseline(gender: str, age: int):
    age_str = get_age_group(age)
    g_code = "남" if gender == "남성" else ("여" if gender == "여성" else None)
    
    if g_code:
        subset = df_stat[(df_stat["성별구분"] == g_code) & (df_stat["5세연령대"] == age_str)]
    else:
        subset = df_stat[df_stat["5세연령대"] == age_str]
        
    if subset.empty:
        return {
            "cancer": 5, "vascular": 5, "injury": 5, "inpatient": 5,
            "recommended_renewal": "비갱신", "c_pct": 5.0, "v_pct": 10.0
        }

    tot = subset["총진료부담액"].sum()
    
    # 96개 세부 행위 키워드 기반 보장군 매핑
    c_mask = subset["국민관심진료행위명"].str.contains("암|종양|위절제|대장절제|식도종양|방사선치료|조혈모세포", na=False)
    v_mask = subset["국민관심진료행위명"].str.contains("관상동맥|대동맥|심장|부정맥|혈관|뇌|신경차단|신경파괴", na=False)
    i_mask = subset["국민관심진료행위명"].str.contains("화상|창상봉합|십자인대|반월판|반월상|인공관절|회전근개|절골술|물리치료|재활치료", na=False)
    inp_mask = subset["의료 서비스"].isin(["입원", "수술"])

    c_ratio = subset[c_mask]["총진료부담액"].sum() / tot
    v_ratio = subset[v_mask]["총진료부담액"].sum() / tot
    i_ratio = subset[i_mask]["총진료부담액"].sum() / tot
    inp_ratio = subset[inp_mask]["총진료부담액"].sum() / tot

    # 전체 데이터 Min-Max 기반 1~10 척도 정규화
    def scale_val(val, min_v, max_v):
        norm = (val - min_v) / (max_v - min_v) if max_v > min_v else 0.5
        return int(max(1, min(round(1 + norm * 9), 10)))

    rec_cancer = scale_val(c_ratio, 0.003, 0.103)
    rec_vasc = scale_val(v_ratio, 0.020, 0.172)
    rec_inj = scale_val(i_ratio, 0.018, 0.170)
    rec_inpatient = scale_val(inp_ratio, 0.020, 0.350)
    rec_renewal = "비갱신" if age < 50 else "갱신"

    return {
        "cancer": rec_cancer,
        "vascular": rec_vasc,
        "injury": rec_inj,
        "inpatient": rec_inpatient,
        "recommended_renewal": rec_renewal,
        "c_pct": round(c_ratio * 100, 1),
        "v_pct": round(v_ratio * 100, 1)
    }

# -------------------------------------------------------------
# 2. UI 레이아웃 및 편차(Delta) 산출
# -------------------------------------------------------------
st.title("🛡️ AHP-AgenaRisk-RAG 지능형 보험 추천 엔진")
st.caption("공공데이터 통계 기준선 + 개인 위험 편차(Delta) + 약관 독소조항 팩트체크 추천")

st.sidebar.header("1️⃣ 기본 정보 설정")
gender = st.sidebar.radio("성별", ["남성", "여성", "미지정"], horizontal=True)
age = st.sidebar.slider("나이", 0, 90, 24)

baseline = get_public_baseline(gender, age)

renewal_pref = st.sidebar.radio(
    f"갱신 유무 선호 (통계 권장: {baseline['recommended_renewal']})",
    ["비갱신", "갱신", "미지정"],
    index=0 if baseline["recommended_renewal"] == "비갱신" else 1,
    horizontal=True
)
royalty = st.sidebar.slider("로열티 (브랜드 선호도)", 0, 100, 67)

st.sidebar.markdown("---")
st.sidebar.header("2️⃣ 세부 보장 집중도 (AHP)")
st.sidebar.info(
    f"💡 **{age}세 {gender}** 공공 통계 기준선이 기본 적용되었습니다.\n\n"
    f"• 진료비 비중: 암 {baseline['c_pct']}%, 뇌·심장 {baseline['v_pct']}%\n"
    f"• 슬라이더를 움직여 개인 편차(가족력/생활위험)를 반영하세요."
)

cancer_val = st.sidebar.slider("암 질환 보장 집중도", 1, 10, baseline["cancer"], 1, key=f"c_{gender}_{age}")
vascular_val = st.sidebar.slider("뇌·심장 질환 보장 집중도", 1, 10, baseline["vascular"], 1, key=f"v_{gender}_{age}")
injury_val = st.sidebar.slider("상해 및 생활 위험 집중도", 1, 10, baseline["injury"], 1, key=f"i_{gender}_{age}")
inpatient_val = st.sidebar.slider("입원/수술 보장 집중도", 1, 10, baseline["inpatient"], 1, key=f"inp_{gender}_{age}")
claim_rate_val = st.sidebar.slider("목표 지급률 (면책 리스크 민감도, %)", 0, 100, 50, 5)

# 통계 기준선 대비 개인 편차(Delta) 산출
deltas = {
    "cancer": cancer_val - baseline["cancer"],
    "vascular": vascular_val - baseline["vascular"],
    "injury": injury_val - baseline["injury"],
    "inpatient": inpatient_val - baseline["inpatient"]
}

active_deltas = [f"{k.upper()} ({v:+d})" for k, v in deltas.items() if v != 0]
if active_deltas:
    st.sidebar.caption(f"🎯 **통계 대비 개인 보정값:** {', '.join(active_deltas)}")
else:
    st.sidebar.caption("🎯 현재 공공데이터 표준 기준선과 동일합니다.")

st.sidebar.markdown("---")
st.sidebar.header("3️⃣ 개인 맞춤 조건 & 독소조항 필터")
selected_quick_tags = st.sidebar.multiselect(
    "자주 찾는 특이사항/기피조건 (빠른 선택)",
    ["이륜차/오토바이 운전", "전이암/유사암 보장 필수", "비갱신형 상품 선호", "감액/면책기간 최소화 희망"],
    default=[]
)
user_notes = st.sidebar.text_area(
    "추가 기저질환 또는 상세 요청사항 입력",
    placeholder="예: 이륜차 운전 중, 전이암 제외된 약관은 피하고 싶음, 갑상선 질환 이력 등"
)

# -------------------------------------------------------------
# 3. 파이프라인 호출 (엔진 실행)
# -------------------------------------------------------------
raw_products = []
if os.path.exists(products_path):
    with open(products_path, "r", encoding="utf-8") as f:
        raw_products = json.load(f)

user_prefs = {
    "cancer_val": cancer_val,
    "vascular_val": vascular_val,
    "injury_val": injury_val,
    "inpatient_val": inpatient_val,
    "baseline": baseline,
    "deltas": deltas,
    "claim_rate_val": claim_rate_val,
    "royalty": royalty,
    "renewal_pref": renewal_pref
}

evaluated_products, validated_risk_keys = run_evaluation_pipeline(
    raw_products=raw_products,
    user_prefs=user_prefs,
    user_notes=user_notes,
    selected_tags=selected_quick_tags
)

# -------------------------------------------------------------
# 4. 결과 화면 및 XAI AI 리포트 렌더링
# -------------------------------------------------------------
if evaluated_products:
    best = evaluated_products[0]

    st.success(f"## 🏆 최종 1순위 최적 추천 상품: **{best['name']}**")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("종합 적합도 점수", f"{best['score_100']} 점", delta=f"S3: {best['s3']:.3f}")
    m2.metric("예상 월 납입 보험료", f"{best['premium']:,} 원")
    m3.metric("보장 특약 수", f"{best['rider_count']} 개 (갱신형 {best['renewable_count']}개)")
    m4.metric("검출된 제약/독소조항", f"{best['toxic_count'] + len(best['llm_toxics'])} 건")

    st.markdown("### 📋 AI 추천 사유 및 약관 정밀 분석 리포트")

    with st.container():
        st.markdown(f"""
        #### 1. 왜 이 상품이 1위인가요?
        * **공공데이터 부합 및 개인 보정:** **{age}세 {gender}** 공공 통계(암 {baseline['c_pct']}%, 뇌·심장 {baseline['v_pct']}%)에 사용자의 개인 집중도 편차가 합산되어 최적 배분되었습니다.
        * **보장 충실도 & 가성비:** 필수 보장 정규화 점수 **{best['amt_score']}점**, 신의료 특약 다양성 **{best['breadth_score']}점**을 획득했으며, 불필요한 과잉 설계로 인한 가격 감점(`price_penalty`: {best['price_penalty']}점)을 최소화했습니다.
        * **대표 주요 보장:** {', '.join(best['highlights']) if best['highlights'] else '기본 보장 구성'} 외 총 {best['rider_count']}개 특약 탑재
        """)

        st.markdown("#### 2. 개인 맞춤 특이사항 및 독소조항 팩트체크")
        if validated_risk_keys:
            st.info(f"🔍 **분석에 반영된 위험 관심 키워드:** `{', '.join(validated_risk_keys)}`")

        if best["llm_toxics"]:
            st.error(f"🚨 **LLM 약관 심층 분석 결과:** 사용자 기피조건 관련 독소조항 {len(best['llm_toxics'])}건이 검출되어 점수가 감점되었습니다.")
            for item in best["llm_toxics"]:
                st.markdown(f"- **[{item.get('target_name', '특약 약관')}]** {item.get('issue_summary')} `[위험도: {item.get('severity', 'MEDIUM')}]`")
        elif best["user_risks"]:
            st.warning(f"⚠️ **1위 추천 상품 주의:** 약관 내 다음 제약 사항이 확인되었습니다:\n- " + "\n- ".join(best["user_risks"]))
        else:
            st.info("✅ **독소조항 안전:** 입력하신 조건에 위배되거나 불리한 치명적인 독소조항이 1위 추천 상품에서는 발견되지 않았습니다.")

        if len(evaluated_products) > 1:
            st.markdown("#### 3. 다른 비교 상품들의 평가 및 순위 하락 이유")
            for other in evaluated_products[1:]:
                reasons = []
                if other.get("llm_toxics"):
                    reasons.append(f"LLM 독소조항 적발({len(other['llm_toxics'])}건)")
                if other["user_risks"]:
                    reasons.append(f"사용자 기피 조건 적발({', '.join(other['user_risks'])})")
                if other["toxic_count"] > best["toxic_count"]:
                    reasons.append(f"약관 내 독소조항 감점({other['toxic_count']}건)")
                if other["price_penalty"] > best["price_penalty"]:
                    reasons.append(f"과도한 월 보험료 부담({other['premium']:,}원)")
                if other["breadth_score"] < best["breadth_score"]:
                    reasons.append(f"치료비/특약 다양성 부족({other['breadth_score']}점)")
                
                reason_str = ", ".join(reasons) if reasons else "종합 점수 열세"
                st.write(f"• **{other['name']} (적합도: {other['score_100']}점 / S3: {other['s3']:.3f})** : {reason_str}")

    st.markdown("---")

    tab1, tab2 = st.tabs(["📊 3단계 시나리오별 점수 비교 차트", "📑 전체 상품 종합 비교표"])

    with tab1:
        names = [p["name"] for p in evaluated_products]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="기존 AHP (Scenario 1)", x=names, y=[p["s1"] for p in evaluated_products], marker_color="#95a5a6"))
        fig.add_trace(go.Bar(name="AHP + 비검증 LLM (Scenario 2)", x=names, y=[p["s2"] for p in evaluated_products], marker_color="#e74c3c"))
        fig.add_trace(go.Bar(name="AHP + RAG 제안 시스템 (Scenario 3)", x=names, y=[p["s3"] for p in evaluated_products], marker_color="#2ecc71"))
        
        fig.update_layout(
            barmode="group",
            title="3단계 알고리즘 평가 점수 비교 (RAG 검증을 통한 순위 역전 시각화)",
            yaxis_title="추천 점수 (Score)",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        df_rank = pd.DataFrame(evaluated_products)[["name", "score_100", "s3", "s1", "s2", "premium", "rider_count", "toxic_count", "price_penalty"]]
        df_rank.columns = ["상품명", "적합도(100점)", "제안시스템(S3)", "기존AHP(S1)", "비검증LLM(S2)", "월 보험료", "특약수", "독소조항 수", "가격감점"]
        st.dataframe(df_rank, use_container_width=True)
else:
    st.error("`products.json`에 상품 데이터가 존재하지 않습니다.")