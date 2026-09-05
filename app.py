import os
import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pipeline import run_evaluation_pipeline

# 충돌 제거를 위해 2. 레이아웃 위치에서 최상위 배치함
st.set_page_config(page_title="AI 지능형 보험 추천 시스템", layout="wide", page_icon="🛡️")

# -------------------------------------------------------------
# 1. 데이터 로드 및 기준선 산출
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
stat_path = os.path.join(BASE_DIR, "데이터 2차 정제 결과.csv")
cat_path = os.path.join(BASE_DIR, "데이터 2차 정제 결과_대분류.csv")
products_path = os.path.join(BASE_DIR, "products.json")

@st.cache_data
def load_public_data():
    try:
        df_stat = pd.read_csv(stat_path, encoding="utf-8")
    except:
        df_stat = pd.read_csv(stat_path, encoding="cp949")
    try:
        df_cat = pd.read_csv(cat_path, encoding="utf-8")
    except:
        df_cat = pd.read_csv(cat_path, encoding="cp949")

    merged = pd.merge(df_stat, df_cat, on=["국민관심진료행위코드", "국민관심진료행위명"], how="left")
    merged["총진료부담액"] = merged["환자수"] * merged["1인당_평균진료금"]
    return merged

df_merged = load_public_data()

def get_age_group(age: int) -> str:
    if age < 5: return "1_4세"
    elif age >= 80: return "80세 이상"
    else:
        lower = (age // 5) * 5
        return f"{lower}_{lower + 4}세"

def get_public_baseline(gender: str, age: int):
    age_str = get_age_group(age)
    if gender in ["남성", "여성"]:
        g_code = "남" if gender == "남성" else "여"
        subset = df_merged[(df_merged["성별구분"] == g_code) & (df_merged["5세연령대"] == age_str)]
    else:
        subset = df_merged[df_merged["5세연령대"] == age_str]

    if subset.empty:
        return {"cancer": 0.25, "vascular": 0.25, "injury": 0.25, "inpatient": 0.25}

    cat_totals = subset.groupby("대분류")["총진료부담액"].sum()
    cancer = cat_totals.get("암", 0)
    vascular = cat_totals.get("순환계 질환", 0) + cat_totals.get("신경계 질환", 0)
    injury = cat_totals.get("상해", 0)
    disease = cat_totals.get("질병", 0)
    total = max(cancer + vascular + injury + disease, 1)

    service_totals = subset.groupby("의료 서비스")["총진료부담액"].sum()
    inpatient_ratio = (service_totals.get("입원", 0) + service_totals.get("수술", 0)) / max(service_totals.sum(), 1)

    return {
        "cancer": round(cancer / total, 3),
        "vascular": round(vascular / total, 3),
        "injury": round(injury / total, 3),
        "inpatient": round(inpatient_ratio, 3)
    }

# -------------------------------------------------------------
# 2. UI 레이아웃
# -------------------------------------------------------------
st.title("🛡️ AHP-AgenaRisk-RAG 지능형 보험 추천 엔진")
st.caption("공공데이터 통계 기반 자동 가이드라인 + 법률 약관 독소조항 팩트체크 추천")

st.sidebar.header("1️⃣ 기본 정보 설정")
gender = st.sidebar.radio("성별", ["남성", "여성", "미지정"], horizontal=True)

#24는 기본 세팅 값
age = st.sidebar.slider("나이", 0, 90, 24)
renewal_pref = st.sidebar.radio("갱신 유무 선호", ["비갱신", "갱신", "미지정"], horizontal=True)

#67은 기본 세팅값
royalty = st.sidebar.slider("로열티 (브랜드 선호도)", 0, 100, 67)

baseline = get_public_baseline(gender, age)

st.sidebar.markdown("---")
st.sidebar.header("2️⃣ 세부 보장 선호도 (AHP)")
st.sidebar.info(f"💡 {age}세 {gender} 공공데이터 기준선이 자동 적용되었습니다.")

def scale_to_10(ratio_val):
    return int(max(1, min(round(float(ratio_val) * 10), 10)))


#우선순위라는 말보다 다른말고 바꾸는게 나을거같음
cancer_val = st.sidebar.slider("암 질환 보장 우선순위", 1, 10, scale_to_10(baseline["cancer"]), 1)
vascular_val = st.sidebar.slider("뇌·심장 질환 보장 우선순위", 1, 10, scale_to_10(baseline["vascular"]), 1)
injury_val = st.sidebar.slider("상해 및 생활 위험 우선순위", 1, 10, scale_to_10(baseline["injury"]), 1)
inpatient_val = st.sidebar.slider("입원/수술 보장 우선순위", 1, 10, scale_to_10(baseline["inpatient"]), 1)
claim_rate_val = st.sidebar.slider("목표 지급률 (면책 리스크 민감도, %)", 0, 100, 50, 5)

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
        * **공공데이터 부합도:** **{age}세 {gender}** 공공 의료 통계상 우선순위가 높은 주요 질환(암 {baseline['cancer']*100:.1f}%, 뇌·심장 {baseline['vascular']*100:.1f}%)에 맞추어 필수 진단비와 치료비가 빈틈없이 구성되었습니다.
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