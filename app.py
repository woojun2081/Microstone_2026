import os
import json
import datetime
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

try:
    df_stat = load_public_data()
except Exception as e:
    df_stat = pd.DataFrame()

def get_age_group(age: int) -> str:
    if age < 5: return "1_4세"
    elif age >= 80: return "80세 이상"
    else:
        lower = (age // 5) * 5
        return f"{lower}_{lower + 4}세"

def get_public_w_base(gender: str, age: int):
    if df_stat.empty:
        return {
            "암": 0.35, "뇌혈관": 0.15, "심장": 0.15, "상해": 0.15, "입원/수술": 0.20
        }, {"cancer": 5, "brain": 5, "heart": 5, "injury": 5, "inpatient": 5}, "비갱신"

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
# 2. UI 레이아웃 및 세션 상태 관리
# -------------------------------------------------------------
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "search"  # "search", "history", "bookmarks", "rider_detail", "toxic_detail"

if "evaluated_products" not in st.session_state:
    st.session_state["evaluated_products"] = []

if "validated_risks" not in st.session_state:
    st.session_state["validated_risks"] = []

if "user_prefs" not in st.session_state:
    st.session_state["user_prefs"] = {}

if "active_notes" not in st.session_state:
    st.session_state["active_notes"] = ""

# 검색 결과 표시 트리거 (보험 검색 화면 안에서 결과 노출 제어)
if "show_search_results" not in st.session_state:
    st.session_state["show_search_results"] = False

# [핵심] 현재 대시보드에서 선택된 순위 인덱스 (0: 1순위, 1: 2순위, ...)
if "selected_rank_idx" not in st.session_state:
    st.session_state["selected_rank_idx"] = 0

# 최근 조회 내역 리스트 (1번 사진용)
if "search_history" not in st.session_state:
    st.session_state["search_history"] = []

# 즐겨찾기 딕셔너리 (2번 사진용: 상품명 -> {상품정보, 검색당시입력조건})
if "bookmarks_map" not in st.session_state:
    st.session_state["bookmarks_map"] = {}

# 상세 페이지 전달용 상품 객체
if "selected_product_for_detail" not in st.session_state:
    st.session_state["selected_product_for_detail"] = None

if "prev_page" not in st.session_state:
    st.session_state["prev_page"] = "search"

st.markdown("""
<style>
    .hero-banner {
        background: linear-gradient(135deg, #00d2ff 0%, #0066ff 100%);
        border-radius: 16px;
        padding: 42px 35px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(0, 102, 255, 0.18);
    }
    .history-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }
    .history-meta-title {
        font-weight: 700;
        color: #4a5568;
        font-size: 13.5px;
        width: 100px;
    }
    .history-meta-val {
        color: #1a202c;
        font-size: 13.5px;
    }
</style>
""", unsafe_allow_html=True)

# [글로벌 상단 헤더] Bo:On 네비게이션
col_logo, col_space, col_menu1, col_menu2, col_menu3 = st.columns([2.5, 3.5, 1.3, 1.6, 1.4])

with col_logo:
    st.markdown("<h2 style='margin:0; color:#0066ff; font-weight:900;'>Bo:On</h2>", unsafe_allow_html=True)

with col_menu1:
    if st.button("🔍 보험 검색", use_container_width=True, type="primary" if st.session_state["current_page"] == "search" else "secondary"):
        st.session_state["current_page"] = "search"
        st.rerun()

with col_menu2:
    hist_cnt = len(st.session_state["search_history"])
    if st.button(f"📊 최근 조회 결과 ({hist_cnt})", use_container_width=True, type="primary" if st.session_state["current_page"] == "history" else "secondary"):
        st.session_state["current_page"] = "history"
        st.rerun()

with col_menu3:
    bm_cnt = len(st.session_state["bookmarks_map"])
    if st.button(f"⭐ 즐겨찾기 ({bm_cnt})", use_container_width=True, type="primary" if st.session_state["current_page"] == "bookmarks" else "secondary"):
        st.session_state["current_page"] = "bookmarks"
        st.rerun()

st.markdown("---")

# -------------------------------------------------------------
# 공통 함수: 추천 결과 대시보드 렌더러 (순위 선택에 따라 거미줄 + 상태바 동적 변환)
# -------------------------------------------------------------
def render_results_dashboard(source_page_name="search"):
    evaluated_products = st.session_state.get("evaluated_products", [])
    user_prefs = st.session_state.get("user_prefs", {})
    validated_risk_keys = st.session_state.get("validated_risks", [])
    w_final = user_prefs.get("w_final", {"암": 0.2, "뇌혈관": 0.2, "심장": 0.2, "상해": 0.2, "입원/수술": 0.2})

    if not evaluated_products:
        return

    top_n = min(len(evaluated_products), 5)
    
    # 순위 인덱스 안전 보정
    current_idx = st.session_state.get("selected_rank_idx", 0)
    if current_idx >= top_n:
        current_idx = 0
        st.session_state["selected_rank_idx"] = 0

    st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin:0; font-weight:800; color:#1a202c;'>AI가 분석한 항목별 보장 수준과 맞춤형 추천 결과</h3>", unsafe_allow_html=True)
    st.caption("공공데이터와 약관 감쇄 엔진을 거쳐 산출된 최적 포트폴리오입니다. 순위 버튼을 누르면 해당 보험의 그래프와 지표로 즉시 전환됩니다.")
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    # [1] 순위 선택 가로 버튼 바 (6번째 사진 탭 스타일 연동)
    rank_cols = st.columns(top_n)
    for r_idx in range(top_n):
        with rank_cols[r_idx]:
            is_active = (r_idx == current_idx)
            btn_label = f"🏆 {r_idx+1}순위" if r_idx == 0 else f"{r_idx+1}순위"
            if st.button(btn_label, key=f"rank_selector_{source_page_name}_{r_idx}", use_container_width=True, type="primary" if is_active else "secondary"):
                st.session_state["selected_rank_idx"] = r_idx
                st.rerun()

    # 현재 선택된 순위의 보험 상품 객체
    selected_prod = evaluated_products[current_idx]

    # 종합 점수 상태 및 신호등 대표 색상 판정
    score_val = selected_prod.get("score_100", 0.0)
    if score_val >= 80.0:
        badge_color = "#28a745"
        badge_text = "🟢 안심 추천 (적합)"
    elif score_val >= 50.0:
        badge_color = "#ffc107"
        badge_text = "🟡 유의 검토 (보통)"
    else:
        badge_color = "#dc3545"
        badge_text = "🔴 가입 재고 (위험)"

    # [방사형 레이더(거미줄) 차트 생성 함수]
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
                radialaxis=dict(visible=True, range=[0, 100], tickvals=[20, 40, 60, 80, 100], ticktext=["20", "40", "60", "80", "100"], tickfont=dict(size=9, color="#888888"), gridcolor="#e2e8f0"),
                angularaxis=dict(tickfont=dict(size=12, color="#2d3748", family="-apple-system, sans-serif"), gridcolor="#e2e8f0", rotation=90, direction="clockwise"),
                bgcolor="rgba(0,0,0,0)"
            ),
            showlegend=False,
            height=260,
            margin=dict(l=40, r=40, t=25, b=25),
            paper_bgcolor="rgba(0,0,0,0)"
        )
        return fig

    # [2] 상단 종합 카드: 선택된 순위 보험의 100점 점수 + 방사형 거미줄 차트
    with st.container(border=True):
        col_summary, col_radar = st.columns([1.1, 1.2])

        with col_summary:
            st.caption(f"🛡️ **[{current_idx+1}순위] AI 종합 적합도 및 핵심 지표**")
            st.markdown(f"<h4 style='margin:2px 0 8px 0; color:#1a202c; font-weight:800;'>{selected_prod['name']}</h4>", unsafe_allow_html=True)
            st.markdown(f"""
                <div style="margin-top: 5px; margin-bottom: 12px;">
                    <div style="font-size: 38px; font-weight: 800; color: {badge_color}; line-height: 1.1;">
                        {score_val}점 <span style="font-size: 20px; color: #718096; font-weight: 600;">/ 100점</span>
                    </div>
                    <div style="margin-top: 6px; font-size: 15px; font-weight: 700; color: {badge_color};">
                        {badge_text}
                    </div>
                    <div style="font-size: 12px; color: #a0aec0; margin-top: 2px;">
                        (약관 감쇄 모델 S3: {selected_prod.get('s3', 0.0):.3f})
                    </div>
                </div>
            """, unsafe_allow_html=True)

            sub_m1, sub_m2 = st.columns(2)
            with sub_m1:
                prem_won = selected_prod['premium']
                prem_display = f"{prem_won // 10000}만 {prem_won % 10000:,}원" if prem_won >= 10000 else f"{prem_won:,}원"
                st.metric("월 납입 보험료", prem_display)
            with sub_m2:
                st.metric("보장 특약 수", f"{selected_prod['rider_count']} 개")

            total_toxic_count = selected_prod.get("toxic_count", 0) + len(selected_prod.get("llm_toxics", []))
            if total_toxic_count > 0:
                with st.popover(f"⚠️ 독소/면책 탐지: **{total_toxic_count}건** (상세 확인 🔍)", use_container_width=True):
                    st.markdown("#### 🔍 탐지된 독소 및 면책 조항 상세")
                    st.caption("약관 분석 엔진이 해당 상품에서 감지한 제약 및 면책 조건입니다.")
                    st.divider()

                    has_content = False
                    if selected_prod.get("llm_toxics"):
                        has_content = True
                        st.markdown("**🤖 AI 정밀 분석 독소조항**")
                        for item in selected_prod["llm_toxics"]:
                            st.error(f"**[{item.get('target_name', '특약')}]** {item.get('issue_summary')}")

                    if selected_prod.get("user_risks"):
                        has_content = True
                        st.markdown("**🎯 사용자 입력 기피 조건 매칭**")
                        for r in selected_prod["user_risks"]:
                            st.warning(f"• {r}")

                    inherent_list = selected_prod.get("inherent_toxics", [])
                    if inherent_list:
                        has_content = True
                        st.markdown("**📄 약관 내 면책/감액 조항 포함 특약**")
                        for s_idx, item in enumerate(inherent_list, 1):
                            st.markdown(f"**{s_idx}. {item['rider_name']}**")
                            st.caption(f"↳ 탐지 사유: `{item['reason']}`")

                    if not has_content:
                        st.info(f"약관 내 총 **{total_toxic_count}건**의 면책/감액 키워드가 감지되었습니다.")
            else:
                st.caption("✅ **독소/면책 탐지:** 발견된 위험 조항 없음 (안전)")

        with col_radar:
            st.caption(f"🕸️ **[{current_idx+1}순위] 5대 보장 밸런스 프로필 (방사형 능력치)**")
            selected_cat_scores = selected_prod.get("category_scores", {})
            fig_radar = create_radar_chart(selected_cat_scores, badge_color)
            st.plotly_chart(fig_radar, use_container_width=True)

    # [UI 컴포넌트] 항목별 신호등 일자 바 & 핀 포인트 렌더러
    def render_traffic_light_slider(label: str, score: float, description: str):
        clamped_score = min(max(score, 0.0), 100.0)
        pin_color = "#34a853" if clamped_score >= 70 else ("#fbbc04" if clamped_score >= 40 else "#ea4335")
        status_label = "높음" if clamped_score >= 70 else ("보통" if clamped_score >= 40 else "낮음")

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

    # [3] 선택된 순위 보험의 상세 보장 수준 (신호등 상태바 연동)
    with st.container(border=True):
        st.markdown(f"""
            <div style="text-align: center; margin-bottom: 15px;">
                <h3 style="font-size: 21px; font-weight: 800; color: #1a202c; margin-bottom: 4px;">[{current_idx+1}순위] 선택한 보험의 상세 보장 수준</h3>
                <p style="color: #718096; font-size: 13.5px; margin: 0;"><strong>{selected_prod['name']}</strong>의 5대 항목별 보장 점수 및 분석 내역입니다.</p>
            </div>
        """, unsafe_allow_html=True)

        cat_scores = selected_prod.get("category_scores", {})
        highlights = selected_prod.get("highlights", [])

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

        for label in ["암", "뇌 질환", "심장 질환", "상해", "입원/수술"]:
            sc = cat_scores.get(label, 40.0)
            desc = generate_desc(label, sc)
            render_traffic_light_slider(label, sc, desc)

        st.markdown("""
            <div style="display: flex; justify-content: center; gap: 30px; margin: 20px auto 0 auto; font-size: 12.5px; font-weight: 700;">
                <span style="color: #ea4335;">● 낮음 (0~39점)</span>
                <span style="color: #fbbc04;">● 보통 (40~69점)</span>
                <span style="color: #34a853;">● 높음 (70~100점)</span>
            </div>
        """, unsafe_allow_html=True)

    # [4] 전체 보험 보장 및 독소조항 비교 테이블 (사진 1 반영)
    with st.container(border=True):
        st.markdown("""
            <div style="text-align: center; margin-bottom: 20px;">
                <h3 style="font-size: 21px; font-weight: 800; color: #1a202c; margin-bottom: 4px;">전체 보험 보장 및 독소조항 비교</h3>
                <p style="color: #718096; font-size: 13.5px; margin: 0;">숫자를 클릭하여 세부내용 또는 독소조항을 확인해 보세요.</p>
            </div>
        """, unsafe_allow_html=True)

        th_star, th_name, th_prem, th_riders, th_toxics = st.columns([0.8, 4.2, 1.6, 1.2, 1.2])
        with th_star: st.markdown("**즐겨찾기**")
        with th_name: st.markdown("**보험명**")
        with th_prem: st.markdown("**월보험료**")
        with th_riders: st.markdown("**보장 수**")
        with th_toxics: st.markdown("**독소조항 수**")
        st.markdown("<hr style='margin: 8px 0;'>", unsafe_allow_html=True)

        top_products = evaluated_products[:5]
        for idx, p in enumerate(top_products):
            c_star, c_name, c_prem, c_riders, c_toxics = st.columns([0.8, 4.2, 1.6, 1.2, 1.2])
            p_name = p.get("name", f"보험 상품 {idx+1}")
            is_bm = p_name in st.session_state["bookmarks_map"]

            # 즐겨찾기 별표 토글
            with c_star:
                star_label = "⭐" if is_bm else "☆"
                if st.button(star_label, key=f"star_btn_{source_page_name}_{idx}", help="즐겨찾기 등록/해제"):
                    if is_bm:
                        del st.session_state["bookmarks_map"][p_name]
                    else:
                        st.session_state["bookmarks_map"][p_name] = {
                            "product": p,
                            "snapshot": st.session_state.get("last_search_snapshot", {})
                        }
                    st.rerun()

            with c_name:
                # 선택된 순위는 파란색 볼드 강조 표시
                badge_prefix = f"**[{idx+1}순위]** "
                if idx == current_idx:
                    st.markdown(f":blue[{badge_prefix}{p_name}] *(선택됨)*")
                else:
                    if st.button(f"{badge_prefix}{p_name}", key=f"select_from_table_{source_page_name}_{idx}", help="이 보험의 거미줄 그래프 및 상세 상태바 확인"):
                        st.session_state["selected_rank_idx"] = idx
                        st.rerun()

            with c_prem:
                st.markdown(f"**{p.get('premium', 0):,}원**")

            with c_riders:
                rider_cnt = p.get("rider_count", len(p.get("raw_clauses", [])))
                if st.button(f":blue[**{rider_cnt}개**]", key=f"btn_rider_{source_page_name}_{idx}", help="세부 보장 내역 확인"):
                    st.session_state["selected_product_for_detail"] = p
                    st.session_state["prev_page"] = st.session_state["current_page"]
                    st.session_state["current_page"] = "rider_detail"
                    st.rerun()

            with c_toxics:
                tx_cnt = p.get("toxic_count", 0) + len(p.get("llm_toxics", []))
                tx_color = ":red" if tx_cnt > 0 else ":green"
                if st.button(f"{tx_color}[**{tx_cnt}개**]", key=f"btn_toxic_{source_page_name}_{idx}", help="독소조항 분석 내역 확인"):
                    st.session_state["selected_product_for_detail"] = p
                    st.session_state["prev_page"] = st.session_state["current_page"]
                    st.session_state["current_page"] = "toxic_detail"
                    st.rerun()

            st.markdown("<hr style='margin: 4px 0;'>", unsafe_allow_html=True)

    # [5] 📋 AI 추천 사유 및 약관 분석 리포트
    with st.container(border=True):
        st.markdown("### 📋 AI 추천 사유 및 약관 분석 리포트")
        st.markdown(f"""
        * **보장 가중치 배분:** 편차 보정된 $w_{{final}}$(암 {w_final['암']*100:.1f}%, 뇌 {w_final['뇌혈관']*100:.1f}%, 심장 {w_final['심장']*100:.1f}%, 상해 {w_final['상해']*100:.1f}%, 입원/수술 {w_final['입원/수술']*100:.1f}%) 적용
        * **보장 충실도:** 필수 보장 적합도 **{selected_prod['amt_score']}점**, 특약 다양성 **{selected_prod['breadth_score']}점**
        * **대표 보장 구성:** {', '.join(selected_prod['highlights']) if selected_prod['highlights'] else '기본 보장 탑재'}
        """)

        if st.session_state["active_notes"].strip():
            if validated_risk_keys:
                st.info(f"🔍 **사용자 입력에서 분석된 키워드:** `{', '.join(validated_risk_keys)}`")

            if selected_prod["llm_toxics"] or selected_prod["user_risks"]:
                st.warning("⚠️ **[주의] 현재 추천 상품 약관 내 제약 사항:**\n입력하신 조건과 관련하여 아래 특약에서 면책 또는 지급 제한 가능성이 확인되었습니다:")
                for item in selected_prod["llm_toxics"]:
                    st.markdown(f"- **[{item.get('target_name', '특약 약관')}]** {item.get('issue_summary')}")
                for risk in selected_prod["user_risks"]:
                    st.markdown(f"- {risk}")
            else:
                st.info(f"✅ **독소조항 안전:** 입력하신 기피 조건에 위배되는 독소조항이 [{current_idx+1}순위] 상품에서는 발견되지 않았습니다.")

            dropped_due_to_toxic = [p for p in evaluated_products[1:] if p["llm_toxics"] or p["user_risks"]]
            if dropped_due_to_toxic:
                st.error("🚨 **독소조항 감지로 인한 대체 추천 안내:**")
                seen_base_names = set()
                unique_toxic_prods = []
                for other in dropped_due_to_toxic:
                    base_name = other.get("name", "타 후보 보험").split("(")[0].strip()
                    if base_name not in seen_base_names:
                        seen_base_names.add(base_name)
                        unique_toxic_prods.append((base_name, other))

                for base_p_name, other in unique_toxic_prods[:2]:
                    llm_reasons = [f"'{t.get('target_name')}' 약관의 면책 사유({t.get('issue_summary')})" for t in other["llm_toxics"]]
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
                        sample_names = ", ".join(r_list[:2])
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
        else:
            st.caption("💡 1단계 검색 화면에서 기피 조건을 입력하고 검색하시면, 약관 내 면책·독소조항을 RAG-LLM으로 검증하여 대체 상품 추천 사유를 안내합니다.")


# =============================================================
# PAGE 1: 메인 & 보험 검색 페이지 (상하 스크롤 흐름)
# =============================================================
if st.session_state["current_page"] == "search":

    # 1. Hero Section (1번째 사진 반영)
    st.markdown("""
        <div class="hero-banner">
            <h1 style="font-size: 38px; font-weight: 900; margin: 0 0 8px 0; color: #ffffff;">Bo:On</h1>
            <p style="font-size: 18px; font-weight: 500; margin: 0 0 4px 0; color: #ffffff;">보험을 쉽고 똑똑하게</p>
            <p style="font-size: 15px; font-weight: 400; opacity: 0.9; margin: 0; color: #ffffff;">온 가족에게 맞는 최적의 선택을 찾아드립니다.</p>
        </div>
    """, unsafe_allow_html=True)

    # 2. 1️⃣ 기본 정보 설정 카드 (2번째 사진 반영: 년/월/일 분리 입력)
    with st.container(border=True):
        st.markdown("<h4 style='font-weight:800; margin-bottom:2px;'>내 조건에 가장 잘 맞는 최적의 보험 상품을 찾아 드려요</h4>", unsafe_allow_html=True)
        st.caption("기본 인적사항을 입력하시면 국가 통계 기준 최적 보장선이 자동으로 계산됩니다.")
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        r1_c1, r1_c2 = st.columns([1, 1.2])
        with r1_c1:
            gender = st.radio("성별", ["남성", "여성", "미지정"], horizontal=True, key="search_gender")
        with r1_c2:
            renewal_pref = st.radio("갱신 유형", ["갱신", "비갱신", "미지정"], index=1, horizontal=True, key="search_renewal")

        st.markdown("<div style='font-size:14px; font-weight:600; color:#4a5568; margin-top:10px; margin-bottom:4px;'>생년월일</div>", unsafe_allow_html=True)
        c_y, c_m, c_d = st.columns([1.5, 1, 1])
        with c_y:
            b_year = st.number_input("년도", min_value=1930, max_value=2026, value=1995, step=1, label_visibility="collapsed")
        with c_m:
            b_month = st.number_input("월", min_value=1, max_value=12, value=8, step=1, label_visibility="collapsed")
        with c_d:
            b_day = st.number_input("일", min_value=1, max_value=31, value=15, step=1, label_visibility="collapsed")

        today = datetime.date.today()
        try:
            birth_dt = datetime.date(b_year, b_month, b_day)
            age = today.year - birth_dt.year - ((today.month, today.day) < (birth_dt.month, birth_dt.day))
            age = max(age, 0)
        except:
            age = max(today.year - b_year, 0)

        st.caption(f"↳ 입력된 생년월일: **{b_year}년 {b_month:02d}월 {b_day:02d}일** (만 나이: **{age}세**)")

        w_base, slider_defaults, recommended_renewal = get_public_w_base(gender, age)

    # 3. 2️⃣ 세부 보장 집중도 (1~10 척도) 카드 (3~4번째 사진 반영)
    with st.container(border=True):
        st.markdown("<h4 style='font-weight:800; margin-bottom:2px;'>세부 보장 집중도 (1~10 척도)</h4>", unsafe_allow_html=True)
        st.caption(f"💡 {age}세 {gender} 통계 기준선 기본 배치")

        def format_delta_text(current_val, base_val):
            diff = current_val - base_val
            if diff > 0: return f":green[▲ +{diff}] (통계 기준선: {base_val})"
            elif diff < 0: return f":red[▼ {diff}] (통계 기준선: {base_val})"
            else: return f":gray[● 기준선과 동일 ({base_val})]"

        c_s1, c_s2 = st.columns(2)
        with c_s1:
            cancer_val = st.slider("암 질환 보장 집중도", 1, 10, slider_defaults["cancer"], 1, key=f"c_{gender}_{age}")
            st.caption(f"↳ {format_delta_text(cancer_val, slider_defaults['cancer'])}")
            
            brain_val = st.slider("뇌혈관 질환 보장 집중도", 1, 10, slider_defaults["brain"], 1, key=f"b_{gender}_{age}")
            st.caption(f"↳ {format_delta_text(brain_val, slider_defaults['brain'])}")

            heart_val = st.slider("심혈관 질환 보장 집중도", 1, 10, slider_defaults["heart"], 1, key=f"h_{gender}_{age}")
            st.caption(f"↳ {format_delta_text(heart_val, slider_defaults['heart'])}")

        with c_s2:
            injury_val = st.slider("상해 및 생활 위험 집중도", 1, 10, slider_defaults["injury"], 1, key=f"i_{gender}_{age}")
            st.caption(f"↳ {format_delta_text(injury_val, slider_defaults['injury'])}")

            inpatient_val = st.slider("입원/수술 보장 집중도", 1, 10, slider_defaults["inpatient"], 1, key=f"inp_{gender}_{age}")
            st.caption(f"↳ {format_delta_text(inpatient_val, slider_defaults['inpatient'])}")

            claim_rate_val = st.slider("목표 지급률 (면책 리스크 민감도, %)", 0, 100, 50, 5, key="claim_rate_slider")

    # 4. 3️⃣ 맞춤 조건 & 독소조항 필터 카드 (4~5번째 사진 반영)
    with st.container(border=True):
        st.markdown("<h4 style='font-weight:800; margin-bottom:2px;'>개인 맞춤 조건 & 약관 독소조항 필터</h4>", unsafe_allow_html=True)
        st.caption("추가 기저질환 또는 피하고 싶은 약관 입력")

        user_notes_input = st.text_area(
            "추가 기저질환 또는 피하고 싶은 약관 입력",
            value=st.session_state.get("active_notes", ""),
            placeholder="예: 가족력이 있어서 암 보장이 탄탄했으면 좋겠고, 월 납입금은 3만 원 이내로 원해요.",
            height=90,
            label_visibility="collapsed"
        )

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        
        # 분석하기 버튼
        if st.button("🚀 AI 맞춤 보험 분석 및 추천 결과 확인하기", use_container_width=True, type="primary"):
            st.session_state["active_notes"] = user_notes_input

            CRITERIA = ["암", "뇌혈관", "심장", "상해", "입원/수술"]
            slider_scores = [cancer_val, brain_val, heart_val, injury_val, inpatient_val]

            matrix = np.ones((5, 5))
            for i in range(5):
                for j in range(5):
                    matrix[i, j] = slider_scores[i] / slider_scores[j]

            w_user = calculate_ahp_weights(matrix, CRITERIA)
            w_final, delta_dict, alpha, D = apply_delta_correction(w_base, w_user, tau=0.65) # 슬라이더 상태바와 관련한 감쇄 해소를 위해 숫자를 키워야함.

            raw_products = []
            if os.path.exists(products_path):
                try:
                    with open(products_path, "r", encoding="utf-8") as f:
                        raw_products = json.load(f)
                except Exception as e:
                    st.error(f"products.json 로드 중 오류 발생: {e}")
                    raw_products = []

            JOB_KEYWORDS = {
                "사무직": ["사무", "회사원", "연구원", "행정", "개발자", "프로그래머", "학생"],
                "현장직": ["현장", "생산", "제조", "건설", "공장", "정비"],
                "운전직": ["운전", "배달", "기사", "택시", "화물", "라이더", "이륜차"],
                "영업/서비스": ["영업", "판매", "서비스", "상담", "외근"],
                "위험직": ["소방", "경찰", "군인", "특수"]
            }

            active_text = st.session_state["active_notes"].strip()
            detected_job = next((standard_job for standard_job, keywords in JOB_KEYWORDS.items() if any(kw in active_text for kw in keywords)), None)

            age_matched = [p for p in raw_products if abs(int(p.get("entry_age", 30) or 30) - age) <= 10]
            candidates = age_matched if len(age_matched) >= 3 else raw_products

            if detected_job:
                job_matched = [p for p in candidates if detected_job in str(p.get("job_category", "")) or any(kw in str(p.get("product_name", "")) for kw in JOB_KEYWORDS[detected_job])]
                final_target_products = job_matched if job_matched else candidates
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
                "slider_steps": {
                    "암": cancer_val - slider_defaults["cancer"],
                    "뇌혈관": brain_val - slider_defaults["brain"],
                    "심장": heart_val - slider_defaults["heart"],
                    "상해": injury_val - slider_defaults["injury"],
                    "입원/수술": inpatient_val - slider_defaults["inpatient"]
                }
            }

            evaluated_prods, validated_risk_keys = run_evaluation_pipeline(
                raw_products=final_target_products,
                user_prefs=user_prefs,
                user_notes=st.session_state["active_notes"],
                selected_tags=[]
            )
            evaluated_prods = diversify_recommendations(evaluated_prods, max_per_company=2)

            st.session_state["evaluated_products"] = evaluated_prods
            st.session_state["validated_risks"] = validated_risk_keys
            st.session_state["user_prefs"] = user_prefs
            st.session_state["show_search_results"] = True
            st.session_state["selected_rank_idx"] = 0  # 검색 시 1순위로 리셋

            snapshot = {
                "gender": gender,
                "birth_str": f"{b_year}.{b_month:02d}.{b_day:02d}",
                "renewal_pref": f"{renewal_pref}형" if renewal_pref != "미지정" else "미지정",
                "slider_scores": {
                    "암": f"{cancer_val:.1f}",
                    "입원/수술": f"{inpatient_val:.1f}",
                    "뇌 질환": f"{brain_val:.1f}",
                    "심장 질환": f"{heart_val:.1f}",
                    "상해": f"{injury_val:.1f}"
                },
                "notes": user_notes_input if user_notes_input.strip() else "특별한 기피 조건 없음",
                "timestamp": datetime.datetime.now().strftime("%Y.%m.%d %H:%M 조회")
            }
            st.session_state["last_search_snapshot"] = snapshot

            st.session_state["search_history"].insert(0, {
                "snapshot": snapshot,
                "evaluated_products": evaluated_prods,
                "validated_risks": validated_risk_keys,
                "user_prefs": user_prefs,
                "active_notes": user_notes_input
            })

            st.rerun()

    if st.session_state.get("show_search_results", False) and st.session_state["evaluated_products"]:
        render_results_dashboard(source_page_name="search")


# =============================================================
# PAGE 2: 최근 조회 결과 (1번 사진 100% 반영)
# =============================================================
elif st.session_state["current_page"] == "history":
    st.caption("최근 조회 내역")
    st.markdown("<h3 style='margin:0; font-weight:800; color:#1a202c;'>이전에 조회했던 분석 결과를 다시 확인해 보세요</h3>", unsafe_allow_html=True)
    st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)

    history_list = st.session_state.get("search_history", [])

    if not history_list:
        with st.container(border=True):
            st.info("아직 조회된 내역이 없습니다. '보험 검색' 탭에서 진단을 먼저 진행해 주세요.")
    else:
        for idx, item in enumerate(history_list):
            snap = item.get("snapshot", {})
            with st.container(border=True):
                c_top1, c_top2 = st.columns([5, 1.5])
                with c_top1:
                    st.markdown(f"<span style='font-weight:700; color:#2d3748; font-size:15px;'>{snap.get('timestamp', '이전 조회')}</span>", unsafe_allow_html=True)
                with c_top2:
                    if st.button("결과 확인 🔍", key=f"view_hist_{idx}", use_container_width=True, type="primary"):
                        st.session_state["evaluated_products"] = item["evaluated_products"]
                        st.session_state["user_prefs"] = item["user_prefs"]
                        st.session_state["validated_risks"] = item["validated_risks"]
                        st.session_state["active_notes"] = item["active_notes"]
                        st.session_state["show_search_results"] = True
                        st.session_state["selected_rank_idx"] = 0
                        st.session_state["current_page"] = "search"
                        st.rerun()

                st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)

                c_lbl1, c_val1 = st.columns([1.5, 8.5])
                with c_lbl1: st.markdown("<span class='history-meta-title'>기본 정보</span>", unsafe_allow_html=True)
                with c_val1: st.markdown(f"<span class='history-meta-val'>{snap.get('gender', '-')} &nbsp;&nbsp;&nbsp;&nbsp; {snap.get('birth_str', '-')} &nbsp;&nbsp;&nbsp;&nbsp; {snap.get('renewal_pref', '-')}</span>", unsafe_allow_html=True)

                scores = snap.get("slider_scores", {})
                scores_str = f"암 {scores.get('암', '0')} &nbsp;&nbsp;&nbsp;&nbsp; 입원/수술 {scores.get('입원/수술', '0')} &nbsp;&nbsp;&nbsp;&nbsp; 뇌 질환 {scores.get('뇌 질환', '0')} &nbsp;&nbsp;&nbsp;&nbsp; 심장 질환 {scores.get('심장 질환', '0')} &nbsp;&nbsp;&nbsp;&nbsp; 상해 {scores.get('상해', '0')}"
                c_lbl2, c_val2 = st.columns([1.5, 8.5])
                with c_lbl2: st.markdown("<span class='history-meta-title'>보장 중요도</span>", unsafe_allow_html=True)
                with c_val2: st.markdown(f"<span class='history-meta-val'>{scores_str}</span>", unsafe_allow_html=True)

                c_lbl3, c_val3 = st.columns([1.5, 8.5])
                with c_lbl3: st.markdown("<span class='history-meta-title'>맞춤 조건</span>", unsafe_allow_html=True)
                with c_val3:
                    st.markdown(f"""
                        <div style="background:#f7fafc; border:1px solid #edf2f7; border-radius:8px; padding:8px 12px; color:#4a5568; font-size:13px;">
                            "{snap.get('notes', '-')}"
                        </div>
                    """, unsafe_allow_html=True)


# =============================================================
# PAGE 3: ⭐ 즐겨찾기 (2번 사진 100% 반영 + 보장수/독소조항수 하이퍼링크 탑재)
# =============================================================
elif st.session_state["current_page"] == "bookmarks":
    st.caption("즐겨찾기")
    st.markdown("<h3 style='margin:0; font-weight:800; color:#1a202c;'>즐겨찾기한 보험 목록과 검색했던 조건을 확인해 보세요</h3>", unsafe_allow_html=True)
    st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)

    bookmarks_map = st.session_state.get("bookmarks_map", {})

    if not bookmarks_map:
        with st.container(border=True):
            st.info("아직 즐겨찾기에 등록된 보험 상품이 없습니다. 비교 테이블에서 별표(☆)를 눌러 등록해 보세요!")
    else:
        for p_name, b_data in list(bookmarks_map.items()):
            prod = b_data.get("product", {})
            snap = b_data.get("snapshot", {})

            with st.container(border=True):
                comp_name = prod.get("company", "보험사")
                st.markdown(f"<span style='font-size:12px; font-weight:700; color:#718096;'>{comp_name}</span>", unsafe_allow_html=True)

                c_bm_title, c_bm_prem, c_bm_riders, c_bm_toxics = st.columns([4.2, 1.8, 1.3, 1.3])
                with c_bm_title:
                    st.markdown(f"⭐ **{p_name}**")
                with c_bm_prem:
                    st.markdown(f"<div style='font-size:12px; color:#a0aec0;'>월 보험료</div><div style='font-size:15px; font-weight:800;'>{prod.get('premium', 0):,}원</div>", unsafe_allow_html=True)
                
                with c_bm_riders:
                    r_cnt = prod.get("rider_count", len(prod.get("raw_clauses", [])))
                    st.markdown("<div style='font-size:12px; color:#a0aec0;'>보장 수</div>", unsafe_allow_html=True)
                    if st.button(f":blue[**{r_cnt}개**]", key=f"btn_bm_rider_{p_name}", help="세부 보장 내역 확인"):
                        st.session_state["selected_product_for_detail"] = prod
                        st.session_state["prev_page"] = "bookmarks"
                        st.session_state["current_page"] = "rider_detail"
                        st.rerun()

                with c_bm_toxics:
                    t_cnt = prod.get("toxic_count", 0) + len(prod.get("llm_toxics", []))
                    tx_color = ":red" if t_cnt > 0 else ":green"
                    st.markdown("<div style='font-size:12px; color:#a0aec0;'>독소조항 수</div>", unsafe_allow_html=True)
                    if st.button(f"{tx_color}[**{t_cnt}개**]", key=f"btn_bm_toxic_{p_name}", help="독소조항 분석 내역 확인"):
                        st.session_state["selected_product_for_detail"] = prod
                        st.session_state["prev_page"] = "bookmarks"
                        st.session_state["current_page"] = "toxic_detail"
                        st.rerun()

                with st.expander("▼ 입력 정보 확인하기", expanded=False):
                    c_lbl1, c_val1 = st.columns([1.5, 8.5])
                    with c_lbl1: st.markdown("<span class='history-meta-title'>기본 정보</span>", unsafe_allow_html=True)
                    with c_val1: st.markdown(f"<span class='history-meta-val'>{snap.get('gender', '-')} &nbsp;&nbsp;&nbsp;&nbsp; {snap.get('birth_str', '-')} &nbsp;&nbsp;&nbsp;&nbsp; {snap.get('renewal_pref', '-')}</span>", unsafe_allow_html=True)

                    scores = snap.get("slider_scores", {})
                    scores_str = f"암 {scores.get('암', '0')} &nbsp;&nbsp;&nbsp;&nbsp; 입원/수술 {scores.get('입원/수술', '0')} &nbsp;&nbsp;&nbsp;&nbsp; 뇌 질환 {scores.get('뇌 질환', '0')} &nbsp;&nbsp;&nbsp;&nbsp; 심장 질환 {scores.get('심장 질환', '0')} &nbsp;&nbsp;&nbsp;&nbsp; 상해 {scores.get('상해', '0')}"
                    c_lbl2, c_val2 = st.columns([1.5, 8.5])
                    with c_lbl2: st.markdown("<span class='history-meta-title'>보장 중요도</span>", unsafe_allow_html=True)
                    with c_val2: st.markdown(f"<span class='history-meta-val'>{scores_str}</span>", unsafe_allow_html=True)

                    c_lbl3, c_val3 = st.columns([1.5, 8.5])
                    with c_lbl3: st.markdown("<span class='history-meta-title'>맞춤 조건</span>", unsafe_allow_html=True)
                    with c_val3:
                        st.markdown(f"""
                            <div style="background:#f7fafc; border:1px solid #edf2f7; border-radius:8px; padding:8px 12px; color:#4a5568; font-size:13px;">
                                "{snap.get('notes', '-')}"
                            </div>
                        """, unsafe_allow_html=True)

                c_sub_b1, c_sub_b2 = st.columns([8, 2])
                with c_sub_b2:
                    if st.button("즐겨찾기 해제", key=f"del_bm_card_{p_name}", use_container_width=True):
                        del st.session_state["bookmarks_map"][p_name]
                        st.rerun()


# =============================================================
# PAGE 4: 선택하신 보험의 세부 보장 내역 (사진 2, 3 상세 뷰)
# =============================================================
elif st.session_state["current_page"] == "rider_detail":
    sel_prod = st.session_state.get("selected_product_for_detail")

    if not sel_prod:
        st.session_state["current_page"] = st.session_state.get("prev_page", "search")
        st.rerun()

    p_name = sel_prod.get("name", "보험 상품")

    c_back1, c_back2 = st.columns([5, 1.5])
    with c_back1:
        st.caption("보장 상세내용")
        st.markdown("<h3 style='margin:0; font-weight:800; color:#1a202c;'>선택하신 보험의 세부 보장 내역을 확인해보세요</h3>", unsafe_allow_html=True)
    with c_back2:
        back_label = "← 즐겨찾기로 돌아가기" if st.session_state.get("prev_page") == "bookmarks" else "← 검색 화면으로 돌아가기"
        if st.button(back_label, use_container_width=True, type="primary"):
            st.session_state["current_page"] = st.session_state.get("prev_page", "search")
            st.rerun()

    st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(f"<h4 style='color:#2b6cb0; font-weight:800; margin-bottom:4px;'>{p_name}</h4>", unsafe_allow_html=True)
        st.caption("각 담보별 가입 금액과 보험료, 보장 기간을 상세히 보여줍니다.")
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        raw_clauses = sel_prod.get("raw_clauses", [])

        t_c1, t_c2, t_c3, t_c4, t_c5 = st.columns([0.8, 4.5, 1.8, 1.4, 2.0])
        with t_c1: st.markdown("**번호**")
        with t_c2: st.markdown("**담보가입현황**")
        with t_c3: st.markdown("**가입금액**")
        with t_c4: st.markdown("**보험료**")
        with t_c5: st.markdown("**납입기간 / 보험기간**")
        st.markdown("<hr style='margin: 6px 0;'>", unsafe_allow_html=True)

        if raw_clauses:
            for idx, c_text in enumerate(raw_clauses, 1):
                col1, col2, col3, col4, col5 = st.columns([0.8, 4.5, 1.8, 1.4, 2.0])
                amt_str = "기본 보장"
                if "1억" in c_text: amt_str = "1억원"
                elif "5천" in c_text or "5,000만" in c_text: amt_str = "5천만원"
                elif "3천" in c_text or "3,000만" in c_text: amt_str = "3천만원"
                elif "2천" in c_text or "2,000만" in c_text: amt_str = "2천만원"
                elif "1천" in c_text or "1,000만" in c_text: amt_str = "1천만원"
                elif "500만" in c_text: amt_str = "500만원"
                elif "300만" in c_text: amt_str = "300만원"
                elif "100만" in c_text: amt_str = "100만원"
                elif "10만" in c_text: amt_str = "10만원"

                prem_val = (sel_prod.get("premium", 35000) // len(raw_clauses)) if raw_clauses else 1500
                period_str = "30년납 100세만기" if "100세" in c_text or "비갱신" in p_name else "20년납 20년만기"

                with col1: st.write(f"{idx}")
                with col2: st.write(f"**{c_text[:35]}**")
                with col3: st.write(amt_str)
                with col4: st.write(f"{prem_val:,}원")
                with col5: st.write(period_str)
                st.markdown("<hr style='margin: 4px 0;'>", unsafe_allow_html=True)
        else:
            st.info("상세 특약 데이터가 없습니다.")

        st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
        s_c1, s_c2 = st.columns([5, 2])
        with s_c1:
            st.markdown("**보장 보험료 합계**")
            st.markdown("**갱신 보험료 합계**")
            st.markdown("**비갱신 보험료 합계**")
        with s_c2:
            st.markdown(f"**{sel_prod.get('premium', 0):,}원**")
            st.markdown("**0원**")
            st.markdown(f"**{sel_prod.get('premium', 0):,}원**")


# =============================================================
# PAGE 5: 약관 내 독소/면책 조항 상세 분석 (사진 4 상세 뷰)
# =============================================================
elif st.session_state["current_page"] == "toxic_detail":
    sel_prod = st.session_state.get("selected_product_for_detail")

    if not sel_prod:
        st.session_state["current_page"] = st.session_state.get("prev_page", "search")
        st.rerun()

    p_name = sel_prod.get("name", "보험 상품")

    c_back1, c_back2 = st.columns([5, 1.5])
    with c_back1:
        st.caption("독소조항 상세 분석")
        st.markdown(f"<h3 style='margin:0; font-weight:800; color:#1a202c;'>{p_name}</h3>", unsafe_allow_html=True)
        st.caption("약관에 숨겨진 불리한 조건(면책, 감액, 축소 등)을 상세히 분석해 드립니다.")
    with c_back2:
        back_label = "← 즐겨찾기로 돌아가기" if st.session_state.get("prev_page") == "bookmarks" else "← 검색 화면으로 돌아가기"
        if st.button(back_label, use_container_width=True, type="primary"):
            st.session_state["current_page"] = st.session_state.get("prev_page", "search")
            st.rerun()

    st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        inherent_list = sel_prod.get("inherent_toxics", [])
        llm_list = sel_prod.get("llm_toxics", [])

        tx_h1, tx_h2, tx_h3, tx_h4 = st.columns([5.5, 1.6, 1.4, 2.0])
        with tx_h1: st.markdown("**담보별 보장내용 및 분석**")
        with tx_h2: st.markdown("**가입금액**")
        with tx_h3: st.markdown("**보험료**")
        with tx_h4: st.markdown("**납입기간 / 보험기간**")
        st.markdown("<hr style='margin: 6px 0;'>", unsafe_allow_html=True)

        has_toxics = False

        if inherent_list:
            has_toxics = True
            for idx, item in enumerate(inherent_list, 1):
                col1, col2, col3, col4 = st.columns([5.5, 1.6, 1.4, 2.0])
                r_name = item.get("rider_name", "특약")
                r_reason = item.get("reason", "면책 조항 포함")
                
                with col1:
                    st.markdown(f"**{idx}. {r_name}**")
                    st.caption(f"약관 분석: {r_name}에 대한 보장 규정 중 제약 조건이 존재합니다.")
                    st.error(f"🚨 **독소조항 분석:** {r_reason}")
                with col2: st.write("1천만원")
                with col3: st.write("1,120원")
                with col4: st.write("30년납 100세만기")
                st.markdown("<hr style='margin: 4px 0;'>", unsafe_allow_html=True)

        if llm_list:
            has_toxics = True
            for item in llm_list:
                col1, col2, col3, col4 = st.columns([5.5, 1.6, 1.4, 2.0])
                t_name = item.get("target_name", "약관 조항")
                t_issue = item.get("issue_summary", "면책 사유 감지")

                with col1:
                    st.markdown(f"**🤖 [AI 정밀 감지] {t_name}**")
                    st.error(f"🚨 **독소조항 분석:** {t_issue}")
                with col2: st.write("-")
                with col3: st.write("-")
                with col4: st.write("약관 제약")
                st.markdown("<hr style='margin: 4px 0;'>", unsafe_allow_html=True)

        if not has_toxics:
            st.success("✅ 해당 상품은 감지된 독소조항이나 불리한 면책 조건이 없는 안전한 상품입니다.")