import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pipeline import run_evaluation_pipeline, calculate_ahp_weights, apply_delta_correction

st.set_page_config(page_title="AI 지능형 보험 추천 시스템", layout="wide", page_icon="🛡️")

# -------------------------------------------------------------
# [개선] 상품명 중복 제거 + 보험사 다양성 보장 랭킹 재정렬 함수
# -------------------------------------------------------------
def diversify_recommendations(products: list, max_per_company: int = 2) -> list:
    company_counts = {}
    seen_product_names = set()  # 중복 상품명 추적용 집합
    diversified = []
    overflow = []

    for p in products:
        p_name = p.get("name", "").strip()

        # 1. 완전 중복 상품 제외 (이미 담긴 상품명은 스킵)
        if p_name in seen_product_names:
            continue
        seen_product_names.add(p_name)

        # 2. 보험사 키 추출 및 기본값 처리
        comp = p.get("company", "기타")

        # 3. 보험사당 최대 허용 개수 체크
        if company_counts.get(comp, 0) < max_per_company:
            company_counts[comp] = company_counts.get(comp, 0) + 1
            diversified.append(p)
        else:
            overflow.append(p)

    return diversified + overflow

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
st.title("🛡️ 통계/사용자 가중치 편차 보정 및 베이지안 네트워크 보완형 AHP 보험 추천 시스템")

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

def format_delta_text(current_val, base_val):
    diff = current_val - base_val
    if diff > 0:
        return f":green[▲ +{diff}] (통계 기준선: {base_val})"
    elif diff < 0:
        return f":red[▼ {diff}] (통계 기준선: {base_val})"
    else:
        return f":gray[● 기준선과 동일 ({base_val})]"

cancer_val = st.sidebar.slider(
    "암 질환 보장 집중도",
    1,
    10,
    slider_defaults["cancer"],
    1,
    key=f"c_{gender}_{age}",
)
st.sidebar.caption(f"↳ {format_delta_text(cancer_val, slider_defaults['cancer'])}")

brain_val = st.sidebar.slider(
    "뇌혈관 질환 보장 집중도",
    1,
    10,
    slider_defaults["brain"],
    1,
    key=f"b_{gender}_{age}",
)
st.sidebar.caption(f"↳ {format_delta_text(brain_val, slider_defaults['brain'])}")

heart_val = st.sidebar.slider(
    "심혈관 질환 보장 집중도",
    1,
    10,
    slider_defaults["heart"],
    1,
    key=f"h_{gender}_{age}",
)
st.sidebar.caption(f"↳ {format_delta_text(heart_val, slider_defaults['heart'])}")

injury_val = st.sidebar.slider(
    "상해 및 생활 위험 집중도",
    1,
    10,
    slider_defaults["injury"],
    1,
    key=f"i_{gender}_{age}",
)
st.sidebar.caption(f"↳ {format_delta_text(injury_val, slider_defaults['injury'])}")

inpatient_val = st.sidebar.slider(
    "입원/수술 보장 집중도",
    1,
    10,
    slider_defaults["inpatient"],
    1,
    key=f"inp_{gender}_{age}",
)
st.sidebar.caption(
    f"↳ {format_delta_text(inpatient_val, slider_defaults['inpatient'])}"
)

CRITERIA = ["암", "뇌혈관", "심장", "상해", "입원/수술"]
slider_scores = [cancer_val, brain_val, heart_val, injury_val, inpatient_val]

matrix = np.ones((5, 5))
for i in range(5):
    for j in range(5):
        matrix[i, j] = slider_scores[i] / slider_scores[j]

w_user = calculate_ahp_weights(matrix, CRITERIA)
w_final, delta_dict, alpha, D = apply_delta_correction(w_base, w_user, tau=0.65) # 슬라이더 상태바와 관련한 감쇄 해소를 위해 숫자를 키워야함.

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
    try:
        with open(products_path, "r", encoding="utf-8") as f:
            raw_products = json.load(f)
    except Exception as e:
        st.error(f"products.json 로드 중 오류 발생: {e}")
        raw_products = []

# 직업 키워드 매핑 사전 정의
JOB_KEYWORDS = {
    "사무직": ["사무", "회사원", "연구원", "행정", "개발자", "프로그래머", "학생"],
    "현장직": ["현장", "생산", "제조", "건설", "공장", "정비"],
    "운전직": ["운전", "배달", "기사", "택시", "화물", "라이더", "이륜차"],
    "영업/서비스": ["영업", "판매", "서비스", "상담", "외근"],
    "위험직": ["소방", "경찰", "군인", "특수"]
}

active_text = st.session_state["active_notes"].strip()
detected_job = None
if active_text:
    for standard_job, keywords in JOB_KEYWORDS.items():
        if any(kw in active_text for kw in keywords):
            detected_job = standard_job
            break

# 1차 필터링: 나이대 매칭 (입력 나이 기준 ±10세 이내 선별)
age_matched = []
for p in raw_products:
    p_age = int(p.get("entry_age", 30) or 30)
    if abs(p_age - age) <= 10:
        age_matched.append(p)

candidates = age_matched if len(age_matched) >= 3 else raw_products

# 2차 필터링: 직업 매칭 (텍스트 입력 시 해당 직업 우선, 미입력 시 전체 대상)
if detected_job:
    job_matched = [
        p for p in candidates
        if detected_job in str(p.get("job_category", "")) or any(kw in str(p.get("product_name", "")) for kw in JOB_KEYWORDS[detected_job])
    ]
    if job_matched:
        final_target_products = job_matched
        st.info(f"💼 입력하신 내용에서 직업군 **[{detected_job}]**(으)로 분석되어 맞춤 상품군을 우선 선별했습니다.")
    else:
        final_target_products = candidates
else:
    final_target_products = candidates

user_prefs = {
    "age": age,
    "gender": gender,
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
    "deltas": delta_dict,
    # [추가] 사용자가 슬라이더를 실제로 몇 눈금 움직였는지(-9 ~ +9) 계산하여 전달
    "slider_steps": {
        "암": cancer_val - slider_defaults["cancer"],
        "뇌혈관": brain_val - slider_defaults["brain"],
        "심장": heart_val - slider_defaults["heart"],
        "상해": injury_val - slider_defaults["injury"],
        "입원/수술": inpatient_val - slider_defaults["inpatient"]
    }
}

# [수정] 결과를 바로 evaluated_products로 수신하여 이름 충돌 및 NameError 방지
evaluated_products, validated_risk_keys = run_evaluation_pipeline(
    raw_products=final_target_products,
    user_prefs=user_prefs,
    user_notes=st.session_state["active_notes"],
    selected_tags=[]
)

# 다양성 필터 실행
evaluated_products = diversify_recommendations(evaluated_products, max_per_company=2)

# -------------------------------------------------------------
# 4. 추천 결과 및 독소조항 대체 리포트
# -------------------------------------------------------------
if evaluated_products:
    best = evaluated_products[0]
    st.success(f"## 🏆 최종 1순위 최적 추천 상품: **{best['name']}**")

    # 1. 종합 점수 상태 및 신호등 대표 색상 판정
    score_val = best.get("score_100", 0.0)
    if score_val >= 80.0:
        badge_color = "#28a745"
        badge_text = "🟢 안심 추천 (적합)"
    elif score_val >= 50.0:
        badge_color = "#ffc107"
        badge_text = "🟡 유의 검토 (보통)"
    else:
        badge_color = "#dc3545"
        badge_text = "🔴 가입 재고 (위험)"

    # -------------------------------------------------------------
    # [방사형 레이더(거미줄) 차트 생성 함수]
    # -------------------------------------------------------------
    def create_radar_chart(cat_scores: dict, color: str):
        categories = ["암", "뇌 질환", "심장 질환", "상해", "입원/수술"]
        values = [cat_scores.get(c, 40.0) for c in categories]
        
        categories_closed = categories + [categories[0]]
        values_closed = values + [values[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill='toself',
            fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.25)",
            line=dict(color=color, width=2.5),
            marker=dict(size=6, color=color),
            hoverinfo='theta+r'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickvals=[20, 40, 60, 80, 100],
                    ticktext=["20", "40", "60", "80", "100"],
                    tickfont=dict(size=9, color="#888888"),
                    gridcolor="#e2e8f0",
                ),
                angularaxis=dict(
                    tickfont=dict(size=12, color="#2d3748", family="-apple-system, sans-serif"),
                    gridcolor="#e2e8f0",
                    rotation=90,
                    direction="clockwise"
                ),
                bgcolor="rgba(0,0,0,0)"
            ),
            showlegend=False,
            height=260,
            margin=dict(l=40, r=40, t=25, b=25),
            paper_bgcolor="rgba(0,0,0,0)"
        )
        return fig

    # -------------------------------------------------------------
    # [상단 종합 카드: 텍스트 강조 + 방사형 거미줄 차트]
    # -------------------------------------------------------------
    st.markdown("""
        <style>
        .radar-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 18px 24px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.04);
            margin-bottom: 25px;
        }
        </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="radar-card">', unsafe_allow_html=True)
        col_summary, col_radar = st.columns([1.1, 1.2])

        with col_summary:
            st.caption("🛡️ **AI 종합 적합도 및 핵심 지표**")
            st.markdown(f"""
                <div style="margin-top: 5px; margin-bottom: 12px;">
                    <div style="font-size: 38px; font-weight: 800; color: {badge_color}; line-height: 1.1;">
                        {score_val}점 <span style="font-size: 20px; color: #718096; font-weight: 600;">/ 100점</span>
                    </div>
                    <div style="margin-top: 6px; font-size: 15px; font-weight: 700; color: {badge_color};">
                        {badge_text}
                    </div>
                    <div style="font-size: 12px; color: #a0aec0; margin-top: 2px;">
                        (약관 감쇄 모델 S3: {best['s3']:.3f})
                    </div>
                </div>
            """, unsafe_allow_html=True)

            sub_m1, sub_m2 = st.columns(2)
            with sub_m1:
                st.metric("월 납입 보험료", f"{best['premium']:,} 원")
            with sub_m2:
                st.metric("보장 특약 수", f"{best['rider_count']} 개")

            # 플로팅 팝업창으로 상세 내역 표시
            total_toxic_count = best.get("toxic_count", 0) + len(best.get("llm_toxics", []))

            if total_toxic_count > 0:
                with st.popover(f"⚠️ 독소/면책 탐지: **{total_toxic_count}건** (상세 확인 🔍)", use_container_width=True):
                    st.markdown("#### 🔍 탐지된 독소 및 면책 조항 상세")
                    st.caption("약관 분석 엔진이 해당 상품에서 감지한 제약 및 면책 조건입니다.")
                    st.divider()

                    has_content = False

                    # 1) LLM 정밀 분석 독소조항
                    if best.get("llm_toxics"):
                        has_content = True
                        st.markdown("**🤖 AI 정밀 분석 독소조항**")
                        for item in best["llm_toxics"]:
                            st.error(f"**[{item.get('target_name', '특약')}]** {item.get('issue_summary')}")

                    # 2) 사용자 기피 키워드 매칭 내역
                    if best.get("user_risks"):
                        has_content = True
                        st.markdown("**🎯 사용자 입력 기피 조건 매칭**")
                        for r in best["user_risks"]:
                            st.warning(f"• {r}")

                    # 3) 약관 내재 면책/감액 특약 목록
                    inherent_list = best.get("inherent_toxics", [])
                    if inherent_list:
                        has_content = True
                        st.markdown("**📄 약관 내 면책/감액 조항 포함 특약**")
                        for idx, item in enumerate(inherent_list, 1):
                            st.markdown(f"**{idx}. {item['rider_name']}**")
                            st.caption(f"↳ 탐지 사유: `{item['reason']}`")

                    # 위 데이터가 아직 캐시에 없거나 비어있을 때의 안내
                    if not has_content:
                        st.info(f"약관 내 총 **{total_toxic_count}건**의 면책/감액 키워드가 감지되었습니다. (상세 내역을 보려면 Streamlit 서버를 재시작해 주세요.)")
            else:
                st.caption("✅ **독소/면책 탐지:** 발견된 위험 조항 없음 (안전)")

        with col_radar:
            st.caption("🕸️ **5대 보장 밸런스 프로필 (방사형 능력치)**")
            best_cat_scores = best.get("category_scores", {})
            fig_radar = create_radar_chart(best_cat_scores, badge_color)
            st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # [UI 컴포넌트] 항목별 신호등 일자 바 & 핀 포인트 렌더러
    # -------------------------------------------------------------
    def render_traffic_light_slider(label: str, score: float, description: str):
        clamped_score = min(max(score, 0.0), 100.0)

        if clamped_score >= 70:
            pin_color = "#34a853"
            status_label = "높음"
        elif clamped_score >= 40:
            pin_color = "#fbbc04"
            status_label = "보통"
        else:
            pin_color = "#ea4335"
            status_label = "낮음"

        html_code = f"""
        <div style="margin-bottom: 22px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px;">
                <span style="font-weight: 700; font-size: 14.5px; color: #2d3748; width: 85px;">{label}</span>
                <div style="position: relative; flex-grow: 1; margin: 0 16px; height: 32px; display: flex; align-items: center;">
                    <span style="position: absolute; left: -16px; font-size: 11px; color: #a0aec0;">0</span>
                    <div style="width: 100%; height: 7px; border-radius: 4px; background: linear-gradient(to right, #ea4335 0%, #ea4335 40%, #fbbc04 40%, #fbbc04 70%, #34a853 70%, #34a853 100%);"></div>
                    <div style="position: absolute; left: {clamped_score}%; transform: translateX(-50%); display: flex; flex-direction: column; align-items: center; z-index: 2;">
                        <span style="font-size: 11.5px; font-weight: 800; color: #1a202c; background: #ffffff; padding: 0px 5px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.25); margin-bottom: 2px;">
                            {int(clamped_score)}
                        </span>
                        <div style="width: 12px; height: 12px; border-radius: 50%; background: #ffffff; border: 3px solid {pin_color}; box-shadow: 0 1px 3px rgba(0,0,0,0.3);"></div>
                    </div>
                    <span style="position: absolute; right: -25px; font-size: 11px; color: #a0aec0;">100</span>
                </div>
                <span style="font-size: 13px; font-weight: 700; color: {pin_color}; width: 35px; text-align: right;">{status_label}</span>
            </div>
            <div style="margin-left: 85px; padding-left: 16px; font-size: 12.5px; color: #4a5568; line-height: 1.45;">
                ↳ {description}
            </div>
        </div>
        """
        st.markdown(html_code, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # [상세 보장 수준 화면] 1순위 ~ 5순위 탭 전환
    # -------------------------------------------------------------
    st.markdown("""
        <div style="text-align: center; margin-top: 20px; margin-bottom: 20px;">
            <h3 style="font-size: 21px; font-weight: 800; color: #1a202c; margin-bottom: 4px;">선택한 보험의 상세 보장 수준</h3>
            <p style="color: #718096; font-size: 13.5px; margin: 0;">각 보험 안에서 항목별로 얼마나 보장하는지 확인하세요.</p>
        </div>
    """, unsafe_allow_html=True)

    top_n = min(len(evaluated_products), 5)
    tab_titles = [f"{i+1}순위" for i in range(top_n)]
    rank_tabs = st.tabs(tab_titles)

    for idx, tab in enumerate(rank_tabs):
        with tab:
            prod = evaluated_products[idx]
            p_name = prod.get("name", "보험 상품")
            score_total = prod.get("score_100", 0.0)
            cat_scores = prod.get("category_scores", {})
            highlights = prod.get("highlights", [])

            st.markdown(f"""
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 22px 24px; margin-bottom: 22px; box-shadow: 0 1px 4px rgba(0,0,0,0.03);">
                    <div style="text-align: center; margin-bottom: 22px;">
                        <h4 style="font-size: 18px; font-weight: 700; color: #2d3748; margin-bottom: 4px;">{p_name}</h4>
                        <div style="font-size: 22px; font-weight: 800; color: #2b6cb0;">
                            종합 점수: {score_total}점 <span style="font-size: 15px; color: #718096; font-weight: 500;">/ 100점</span>
                        </div>
                    </div>
            """, unsafe_allow_html=True)

            def generate_desc(category: str, score: float) -> str:
                key_map = {"암": "암", "뇌 질환": "뇌혈관", "심장 질환": "심장", "상해": "상해", "입원/수술": "입원/수술"}
                user_weight = w_final.get(key_map.get(category, category), 0.20)
                
                matched = [h for h in highlights if (category == "암" and "암" in h) or 
                                                     (category == "뇌 질환" and "뇌" in h) or 
                                                     (category == "심장 질환" and "심" in h) or 
                                                     (category == "상해" and "상해" in h) or 
                                                     (category == "입원/수술" and any(k in h for k in ["입원", "수술"]))]
                cov_text = f"실제 가입금액 {matched[0]}" if matched else "기본 한도 탑재"

                if score >= 70:
                    return f"<b>사용자 선호 가중치 {user_weight*100:.1f}% 반영</b>: {cov_text}으로, 전체 상품군 대비 가성비 및 보장 충실도가 충분하여 <b>높음</b>으로 평가되었습니다."
                elif score >= 40:
                    return f"<b>사용자 선호 가중치 {user_weight*100:.1f}% 반영</b>: {cov_text} 수준으로, 평균적인 위험 커버리지를 충족하여 <b>보통</b>으로 평가되었습니다."
                else:
                    return f"<b>사용자 선호 가중치 {user_weight*100:.1f}% 반영</b>: {cov_text}이나 가입 조건이 제한적이거나 타 상품군 대비 한도가 낮아 <b>낮음</b>으로 평가되었습니다."

            labels = ["암", "뇌 질환", "심장 질환", "상해", "입원/수술"]
            for label in labels:
                sc = cat_scores.get(label, 40.0)
                desc = generate_desc(label, sc)
                render_traffic_light_slider(label, sc, desc)

            st.markdown("""
                <div style="display: flex; justify-content: center; gap: 30px; margin: 20px auto 0 auto; font-size: 12.5px; font-weight: 700;">
                    <span style="color: #ea4335;">● 낮음 (0~39점)</span>
                    <span style="color: #fbbc04;">● 보통 (40~69점)</span>
                    <span style="color: #34a853;">● 높음 (70~100점)</span>
                </div>
                </div>
            """, unsafe_allow_html=True)

    # ---------------------------------------------------------------
    # 📋 AI 추천 사유 및 약관 분석 리포트
    # ----------------------------------------------------------------
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

            if best["llm_toxics"] or best["user_risks"]:
                st.warning(f"⚠️ **[주의] 현재 추천 상품 약관 내 제약 사항:**\n입력하신 조건과 관련하여 아래 특약에서 면책 또는 지급 제한 가능성이 확인되었습니다:")
                for item in best["llm_toxics"]:
                    st.markdown(f"- **[{item.get('target_name', '특약 약관')}]** {item.get('issue_summary')}")
                for risk in best["user_risks"]:
                    st.markdown(f"- {risk}")
            else:
                st.info("✅ **독소조항 안전:** 입력하신 기피 조건에 위배되는 독소조항이 1순위 추천 상품에서는 발견되지 않았습니다.")

# 독소조항 감지로 인해 밀려난 후보들 필터링
            dropped_due_to_toxic = [
                p for p in evaluated_products[1:] 
                if p["llm_toxics"] or p["user_risks"]
            ]
            if dropped_due_to_toxic:
                st.error("🚨 **독소조항 감지로 인한 대체 추천 안내:**")
                
                seen_base_names = set()  # 직업군만 다른 동일 상품 중복 출력 방지용 집합
                displayed_cnt = 0

                for other in dropped_due_to_toxic:
                    full_p_name = other.get("name", "타 후보 보험")
                    base_p_name = full_p_name.split("(")[0].strip()

                    # 이미 안내한 보험 상품이면 건너뜀 (건설/공무원/주방장 중복 방지)
                    if base_p_name in seen_base_names:
                        continue
                    seen_base_names.add(base_p_name)

                    # 1) LLM이 찾은 독소조항 요약
                    llm_reasons = [f"'{t.get('target_name')}' 약관의 면책 사유({t.get('issue_summary')})" for t in other["llm_toxics"]]
                    
                    # 2) 키워드 매칭된 특약들을 키워드별로 묶어서 요약 ("전이암 관련 면책 조건(암진단비Ⅱ 외 12건)")
                    raw_matched = other.get("user_risks", [])
                    grouped_riders = {}
                    for item_str in raw_matched:
                        if item_str.startswith("[") and "]" in item_str:
                            rk_part, r_name = item_str.split("]", 1)
                            rk_key = rk_part.replace("[", "").strip()
                            grouped_riders.setdefault(rk_key, []).append(r_name.strip())
                        else:
                            grouped_riders.setdefault("기타", []).append(item_str)

                    summary_parts = []
                    for rk_key, r_list in grouped_riders.items():
                        sample_names = ", ".join(r_list[:2])  # 대표 특약 최대 2개만 노출
                        extra_cnt = len(r_list) - 2
                        if extra_cnt > 0:
                            summary_parts.append(f"약관 내 **'{rk_key}'** 관련 면책/제약 조건({sample_names} 외 {extra_cnt}건)")
                        else:
                            summary_parts.append(f"약관 내 **'{rk_key}'** 관련 면책/제약 조건({sample_names})")

                    all_reasons = llm_reasons + summary_parts
                    reason_text = " 및 ".join(all_reasons) if all_reasons else "약관 내 면책/감액 조건"

                    st.markdown(
                        f"> • **{base_p_name}** 상품은 {reason_text} 등의 이유로 **보장을 받지 못할 위험**이 있습니다. "
                        f"따라서 해당 위험이 없는 상품을 대신 1순위로 추천합니다."
                    )

                    displayed_cnt += 1
                    if displayed_cnt >= 2:  # 최대 2개 상품까지만 깔끔하게 비교 안내
                        break
        else:
            st.caption("💡 사이드바 3번에 기피 조건을 적고 **[🔍 독소조항 분석하기]** 버튼을 누르면, 약관 내 면책·독소조항을 RAG-LLM으로 검증하여 대체 상품 추천 사유를 안내합니다.")