import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pipeline import run_evaluation_pipeline, calculate_ahp_weights, apply_delta_correction

st.set_page_config(page_title="AI 지능형 보험 추천 시스템", layout="wide", page_icon="🛡️")

# -------------------------------------------------------------
# 1. 공공데이터 로드 및 1단계 w_base 산출
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

def get_public_w_base(gender: str, age: int):
    age_str = get_age_group(age)
    g_code = "남" if gender == "남성" else ("여" if gender == "여성" else None)
    
    if g_code:
        subset = df_stat[(df_stat["성별구분"] == g_code) & (df_stat["5세연령대"] == age_str)]
    else:
        subset = df_stat[df_stat["5세연령대"] == age_str]
        
    if subset.empty:
        return {
            "암": 0.35, "뇌혈관": 0.15, "심장": 0.15, "상해": 0.15, "입원/수술": 0.20
        }, {"cancer": 5, "brain": 5, "heart": 5, "injury": 5, "inpatient": 5}, "비갱신"

    tot = subset["총진료부담액"].sum()
    if tot == 0: tot = 1.0

    c_mask = subset["국민관심진료행위명"].str.contains("암|종양|위절제|대장절제|식도종양|방사선치료|조혈모세포", na=False)
    b_mask = subset["국민관심진료행위명"].str.contains("뇌|뇌경색|뇌출혈|뇌동맥|신경차단|신경파괴", na=False)
    h_mask = subset["국민관심진료행위명"].str.contains("관상동맥|대동맥|심장|부정맥|혈관|협심증|심근경색", na=False)
    i_mask = subset["국민관심진료행위명"].str.contains("화상|창상봉합|십자인대|반월판|반월상|인공관절|회전근개|절골술|골절", na=False)
    inp_mask = subset["의료 서비스"].isin(["입원", "수술"])

    c_amt = subset[c_mask]["총진료부담액"].sum()
    b_amt = subset[b_mask]["총진료부담액"].sum()
    h_amt = subset[h_mask]["총진료부담액"].sum()
    i_amt = subset[i_mask]["총진료부담액"].sum()
    inp_amt = subset[inp_mask]["총진료부담액"].sum()

    tot_target = c_amt + b_amt + h_amt + i_amt + inp_amt
    if tot_target == 0: tot_target = 1.0

    w_base = {
        "암": round(float(c_amt / tot_target), 4),
        "뇌혈관": round(float(b_amt / tot_target), 4),
        "심장": round(float(h_amt / tot_target), 4),
        "상해": round(float(i_amt / tot_target), 4),
        "입원/수술": round(float(inp_amt / tot_target), 4)
    }

    def scale_val(val, min_v, max_v):
        norm = (val - min_v) / (max_v - min_v) if max_v > min_v else 0.5
        return int(max(1, min(round(1 + norm * 9), 10)))

    slider_defaults = {
        "cancer": scale_val(c_amt / tot, 0.003, 0.103),
        "brain": scale_val(b_amt / tot, 0.010, 0.100),
        "heart": scale_val(h_amt / tot, 0.010, 0.100),
        "injury": scale_val(i_amt / tot, 0.018, 0.170),
        "inpatient": scale_val(inp_amt / tot, 0.020, 0.350)
    }

    rec_renewal = "비갱신" if age < 50 else "갱신"
    return w_base, slider_defaults, rec_renewal

# -------------------------------------------------------------
# 2. UI 레이아웃
# -------------------------------------------------------------
st.title("🛡️ AHP-AgenaRisk-RAG를 활용한 사용자 지원형 보험 추천 시스템")
st.caption("공공데이터 기준선(w_base) + 슬라이더 쌍대비교(w_user) + 편차보정(w_final) + 약관 독소조항 팩트체크")

st.sidebar.header("1️⃣ 기본 정보 설정")
gender = st.sidebar.radio("성별", ["여성", "남성", "미지정"], horizontal=True)
age = st.sidebar.slider("나이", 0, 90, 40)

w_base, slider_defaults, recommended_renewal = get_public_w_base(gender, age)

renewal_pref = st.sidebar.radio(
    f"갱신 유무 선호 (통계 권장: {recommended_renewal})",
    ["비갱신", "갱신", "미지정"],
    index=0 if recommended_renewal == "비갱신" else 1,
    horizontal=True
)

st.sidebar.markdown("---")
st.sidebar.header("2️⃣ 세부 보장 집중도 (1~10 척도)")
st.sidebar.caption(f"💡 {age}세 {gender} 통계 기준선 기본 배치")

cancer_val = st.sidebar.slider("암 질환 보장 집중도", 1, 10, slider_defaults["cancer"], 1, key=f"c_{gender}_{age}")
brain_val = st.sidebar.slider("뇌혈관 질환 보장 집중도", 1, 10, slider_defaults["brain"], 1, key=f"b_{gender}_{age}")
heart_val = st.sidebar.slider("심혈관 질환 보장 집중도", 1, 10, slider_defaults["heart"], 1, key=f"h_{gender}_{age}")
injury_val = st.sidebar.slider("상해 및 생활 위험 집중도", 1, 10, slider_defaults["injury"], 1, key=f"i_{gender}_{age}")
inpatient_val = st.sidebar.slider("입원/수술 보장 집중도", 1, 10, slider_defaults["inpatient"], 1, key=f"inp_{gender}_{age}")

CRITERIA = ["암", "뇌혈관", "심장", "상해", "입원/수술"]
slider_scores = [cancer_val, brain_val, heart_val, injury_val, inpatient_val]

matrix = np.ones((5, 5))
for i in range(5):
    for j in range(5):
        matrix[i, j] = slider_scores[i] / slider_scores[j]

w_user = calculate_ahp_weights(matrix, CRITERIA)
w_final, delta_dict, alpha, D = apply_delta_correction(w_base, w_user, tau=0.3)

claim_rate_val = st.sidebar.slider("목표 지급률 (면책 리스크 민감도, %)", 0, 100, 50, 5)

st.sidebar.markdown("---")
st.sidebar.header("3️⃣ 맞춤 조건 & 독소조항 필터")
user_notes_input = st.sidebar.text_area(
    "추가 기저질환 또는 피하고 싶은 약관 입력",
    placeholder="예: 이륜차 운전 중, 전이암 제외 약관 기피, 감액기간 최소화 희망 등",
    height=100
)

# 세션 상태 초기화
if "active_notes" not in st.session_state:
    st.session_state["active_notes"] = ""

# 분석하기 버튼
if st.sidebar.button("🔍 독소조항 분석하기", use_container_width=True, type="primary"):
    st.session_state["active_notes"] = user_notes_input

# -------------------------------------------------------------
# 3. 파이프라인 호출
# -------------------------------------------------------------
raw_products = []
if os.path.exists(products_path):
    with open(products_path, "r", encoding="utf-8") as f:
        raw_products = json.load(f)

user_prefs = {
    "w_final": w_final,
    "w_base": w_base,
    "w_user": w_user,
    "D": D,
    "alpha": alpha,
    "claim_rate_val": claim_rate_val,
    "renewal_pref": renewal_pref,
    "cancer_val": cancer_val,
    "brain_val": brain_val,
    "heart_val": heart_val,
    "injury_val": injury_val,
    "inpatient_val": inpatient_val,
    "deltas": delta_dict
}

evaluated_products, validated_risk_keys = run_evaluation_pipeline(
    raw_products=raw_products,
    user_prefs=user_prefs,
    user_notes=st.session_state["active_notes"],
    selected_tags=[]
)

# -------------------------------------------------------------
# 4. 가중치 편차 보정 테이블 렌더링
# -------------------------------------------------------------
st.markdown("### 📊 3단계 5대 영역 가중치 편차 보정 결과 ($w_{final}$)")
col_w1, col_w2 = st.columns([3, 2])

with col_w1:
    df_weights = pd.DataFrame({
        "기준": CRITERIA,
        "슬라이더(입력)": slider_scores,
        "w_base (통계)": [f"{w_base[k]:.4f}" for k in CRITERIA],
        "w_user (쌍대비교)": [f"{w_user[k]:.4f}" for k in CRITERIA],
        "Δ (편차)": [f"{delta_dict[k]:+.4f}" for k in CRITERIA],
        "w_final (최종반영)": [f"{w_final[k]:.4f}" for k in CRITERIA]
    })
    st.table(df_weights)

with col_w2:
    st.info(f"""
    * **총 편차 합계 ($D$):** `{D:.4f}`
    * **허용치 ($\tau$):** `0.3`
    * **보정 계수 ($\alpha$):** `min(1, 0.3 / {D:.4f}) = {alpha:.4f}`
    * *통계적 근거($w_{{base}}$)를 기반으로 사용자의 가중치 선호도가 허용치 내에서 안정적으로 보정 반영되었습니다.*
    """)

# -------------------------------------------------------------
# 5. 추천 결과 및 독소조항 대체 리포트
# -------------------------------------------------------------
if evaluated_products:
    best = evaluated_products[0]
    st.success(f"## 🏆 최종 1순위 최적 추천 상품: **{best['name']}**")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("종합 적합도 점수", f"{best['score_100']} 점", delta=f"S3: {best['s3']:.3f}")
    m2.metric("예상 월 납입 보험료", f"{best['premium']:,} 원")
    m3.metric("보장 특약 수", f"{best['rider_count']} 개")
    m4.metric("검출된 제약/독소조항", f"{best['toxic_count'] + len(best['llm_toxics'])} 건")

    st.markdown("### 📋 AI 추천 사유 및 약관 분석 리포트")
    with st.container():
        st.markdown(f"""
        * **보장 가중치 배분:** 편차 보정된 $w_{{final}}$(암 {w_final['암']*100:.1f}%, 뇌 {w_final['뇌혈관']*100:.1f}%, 심장 {w_final['심장']*100:.1f}%, 상해 {w_final['상해']*100:.1f}%, 입원/수술 {w_final['입원/수술']*100:.1f}%) 적용
        * **보장 충실도:** 필수 보장 적합도 **{best['amt_score']}점**, 특약 다양성 **{best['breadth_score']}점**
        * **대표 보장 구성:** {', '.join(best['highlights']) if best['highlights'] else '기본 보장 탑재'}
        """)

        # 독소조항 분석 리포트 분기
        if st.session_state["active_notes"].strip():
            if validated_risk_keys:
                st.info(f"🔍 **사용자 입력에서 분석된 키워드:** `{', '.join(validated_risk_keys)}`")

            # 1위 상품 자체에 제약이 있는 경우
            if best["llm_toxics"] or best["user_risks"]:
                st.warning(f"⚠️ **[주의] 현재 추천 상품 약관 내 제약 사항:**\n입력하신 조건과 관련하여 아래 특약에서 면책 또는 지급 제한 가능성이 확인되었습니다:")
                for item in best["llm_toxics"]:
                    st.markdown(f"- **[{item.get('target_name', '특약 약관')}]** {item.get('issue_summary')}")
                for risk in best["user_risks"]:
                    st.markdown(f"- {risk}")
            else:
                st.info("✅ **독소조항 안전:** 입력하신 기피 조건에 위배되는 독소조항이 1순위 추천 상품에서는 발견되지 않았습니다.")

            # 하위 상품 중 독소조항으로 순위가 밀린 상품 안내 (대체 추천)
            dropped_due_to_toxic = [
                p for p in evaluated_products[1:] 
                if p["llm_toxics"] or p["user_risks"]
            ]
            if dropped_due_to_toxic:
                st.error("🚨 **독소조항 감지로 인한 대체 추천 안내:**")
                for other in dropped_due_to_toxic:
                    reasons = []
                    for t in other["llm_toxics"]:
                        reasons.append(f"'{t.get('target_name')}' 약관의 면책 사유({t.get('issue_summary')})")
                    for r in other["user_risks"]:
                        reasons.append(r)
                    st.markdown(f"> **{other['name']}** 상품은 약관상 **{', '.join(reasons)}** 등의 이유로 **보장을 받지 못할 위험**이 있습니다. 따라서 해당 위험이 없는 상품을 대신 1순위로 추천합니다.")
        else:
            st.caption("💡 사이드바 3번에 기피 조건을 적고 **[🔍 독소조항 분석하기]** 버튼을 누르면, 약관 내 면책·독소조항을 RAG-LLM으로 검증하여 대체 상품 추천 사유를 안내합니다.")

    st.markdown("---")
    tab1, tab2 = st.tabs(["📊 3단계 시나리오 점수 비교", "📑 전체 상품 종합 비교표"])

    with tab1:
        names = [p["name"] for p in evaluated_products]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="기존 AHP (Scenario 1)", x=names, y=[p["s1"] for p in evaluated_products], marker_color="#95a5a6"))
        fig.add_trace(go.Bar(name="AHP + 비검증 LLM (Scenario 2)", x=names, y=[p["s2"] for p in evaluated_products], marker_color="#e74c3c"))
        fig.add_trace(go.Bar(name="AHP + RAG 제안 시스템 (Scenario 3)", x=names, y=[p["s3"] for p in evaluated_products], marker_color="#2ecc71"))
        fig.update_layout(barmode="group", height=430, yaxis_title="추천 점수")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        df_rank = pd.DataFrame(evaluated_products)[["name", "score_100", "s3", "s1", "s2", "premium", "rider_count", "toxic_count"]]
        df_rank.columns = ["상품명", "적합도(100점)", "제안시스템(S3)", "기존AHP(S1)", "비검증LLM(S2)", "월 보험료", "특약수", "독소조항 수"]
        st.dataframe(df_rank, use_container_width=True)