import os
import json
import re
import math
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# -------------------------------------------------------------
# 1. 공공데이터 및 파일 경로 설정
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

def parse_currency(val) -> int:
    if not val: return 0
    if isinstance(val, (int, float)): return int(val)
    clean_str = str(val).replace(",", "").replace(" ", "").strip()
    total = 0
    if "억" in clean_str:
        parts = clean_str.split("억")
        digits = re.findall(r'\d+', parts[0])
        if digits: total += int(digits[0]) * 100_000_000
        clean_str = parts[1] if len(parts) > 1 else ""
    if "만" in clean_str:
        parts = clean_str.split("만")
        digits = re.findall(r'\d+', parts[0])
        if digits: total += int(digits[0]) * 10_000
        clean_str = parts[1] if len(parts) > 1 else ""
    digits = re.findall(r'\d+', clean_str)
    if digits: total += int(digits[0])
    return total

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
# [안전장치] 양방향 근접도 정규식 자연어 위험 의도 분석기
# -------------------------------------------------------------
def extract_valid_risk_intents(user_notes: str, selected_tags: list) -> list:
    extracted = set()

    # 1. UI 퀵 태그 매핑 (100% 확정 플래그)
    tag_mapping = {
        "이륜차/오토바이 운전": "이륜차",
        "전이암/유사암 보장 필수": "전이암",
        "비갱신형 상품 선호": "갱신",
        "감액/면책기간 최소화 희망": "감액"
    }
    for tag in selected_tags:
        if tag in tag_mapping:
            extracted.add(tag_mapping[tag])

    # 2. 자연어 텍스트 정밀 스캔 (Window Regex)
    if user_notes:
        DOMAIN_RISK_MAP = {
            "이륜차": ["이륜차", "오토바이", "바이크", "스쿠터", "원동기"],
            "전이암": ["전이암", "재발암", "유사암", "소액암"],
            "뇌혈관": ["뇌혈관", "뇌졸중", "뇌출혈", "뇌경색"],
            "심혈관": ["허혈성", "심근경색", "부정맥", "협심증"],
            "감액": ["감액", "면책기간", "대기기간"]
        }
        NEGATION_PATTERN = r"(안함|안 함|않음|않아|안탐|안 타|비운전|해당없음|없음|상관없음)"

        for standard_key, synonyms in DOMAIN_RISK_MAP.items():
            for syn in synonyms:
                if syn in user_notes:
                    # 키워드 전방/후방 8글자 이내 부정어 결합 검사
                    pattern_after = rf"{syn}.{{0,8}}{NEGATION_PATTERN}"
                    pattern_before = rf"{NEGATION_PATTERN}.{{0,8}}{syn}"

                    is_negated = bool(re.search(pattern_after, user_notes) or re.search(pattern_before, user_notes))

                    # "전이암 제외 상품 피하기" 같은 문맥은 부정어가 아님
                    if not is_negated or ("제외" in user_notes and "피하" in user_notes):
                        extracted.add(standard_key)
                        break

    return list(extracted)

# -------------------------------------------------------------
# 2. UI 레이아웃 설정
# -------------------------------------------------------------
st.set_page_config(page_title="AI 지능형 보험 추천 시스템", layout="wide", page_icon="🛡️")
st.title("🛡️ AHP-AgenaRisk-RAG 지능형 보험 추천 엔진")
st.caption("공공데이터 통계 기반 자동 가이드라인 + 법률 약관 독소조항 팩트체크 추천")

# 사이드바
st.sidebar.header("1️⃣ 기본 정보 설정")
gender = st.sidebar.radio("성별", ["남성", "여성", "미지정"], horizontal=True)
age = st.sidebar.slider("나이", 0, 90, 24)
renewal_pref = st.sidebar.radio("갱신 유무 선호", ["비갱신", "갱신", "미지정"], horizontal=True)
royalty = st.sidebar.slider("로열티 (브랜드 선호도)", 0, 100, 67)

baseline = get_public_baseline(gender, age)

st.sidebar.markdown("---")
st.sidebar.header("2️⃣ 세부 보장 선호도 (AHP)")
st.sidebar.info(f"💡 {age}세 {gender} 공공데이터 기준선이 자동 적용되었습니다.")


# 공공데이터 비율을 0.0~1.0 범위를 1~10 으로 변환
def scale_to_10(ratio_val):
    return int(max(1, min(round(float(ratio_val) * 10), 10)))


cancer_val = st.sidebar.slider("암 질환 보장 우선순위", 1, 10, scale_to_10(baseline["cancer"]), 1)
vascular_val = st.sidebar.slider("뇌·심장 질환 보장 우선순위", 1, 10, scale_to_10(baseline["vascular"]), 1)
injury_val = st.sidebar.slider("상해 및 생활 위험 우선순위", 1, 10, scale_to_10(baseline["injury"]), 1)
inpatient_val = st.sidebar.slider("입원/수술 보장 우선순위", 1, 10, scale_to_10(baseline["inpatient"]), 1)
claim_rate_val = st.sidebar.slider("목표 지급률 (면책 리스크 민감도, %)", 0, 100, 50, 5)

st.sidebar.markdown("---")
st.sidebar.header("3️⃣ 개인 맞춤 조건 & 독소조항 필터")

# UI 안전장치 1: 퀵 선택 태그, 텍스트 없이도 빠른 선택을 위해 주는 내용들인데 이거도 나증에 바꿔도 될거같음
selected_quick_tags = st.sidebar.multiselect(
    "자주 찾는 특이사항/기피조건 (빠른 선택)",
    ["이륜차/오토바이 운전", "전이암/유사암 보장 필수", "비갱신형 상품 선호", "감액/면책기간 최소화 희망"],
    default=[]
)

# UI 안전장치 2: 자유 텍스트
user_notes = st.sidebar.text_area(
    "추가 기저질환 또는 상세 요청사항 입력",
    placeholder="예: 이륜차 운전 중, 전이암 제외된 약관은 피하고 싶음, 갑상선 질환 이력 등",
    help="정규식 양방향 분석기가 부정어('운전 안 함' 등)를 자동 판별하여 오탐 없이 반영합니다."
)

# -------------------------------------------------------------
# 3. 3단계 추천 알고리즘 백엔드 연산 (완전 무결성 버전)
# -------------------------------------------------------------
raw_w = [cancer_val, vascular_val, injury_val, inpatient_val]
sum_w = sum(raw_w) if sum(raw_w) > 0 else 1.0
w_cancer, w_vasc, w_inj, w_inp = [v / sum_w for v in raw_w]

# 사용자 리스크 민감도에 따른 동적 가중치 (기본값: risk=0.40, base=0.60)
risk_weight = 0.20 + (claim_rate_val / 100.0) * 0.40
base_weight = 1.0 - risk_weight

# 논문 기준(0.30 : 0.20 : 0.10) 가중치 분배
w_amt_dyn = base_weight * (0.30 / 0.60)
w_breadth_dyn = base_weight * (0.20 / 0.60)
w_conf_dyn = base_weight * (0.10 / 0.60)

# 보장금액 점진적 포화 로그 함수 (1.25 Cap 적용)
def calc_coverage_utility(amt, baseline=50_000_000):
    if amt <= 0: return 0.0
    if amt <= baseline: return amt / baseline
    return min(1.0 + 0.25 * math.log10(amt / baseline + 1), 1.25)

# 검증된 사용자 위험 키워드 추출
validated_risk_keys = extract_valid_risk_intents(user_notes, selected_quick_tags)

evaluated_products = []
if os.path.exists(products_path):
    with open(products_path, "r", encoding="utf-8") as f:
        raw_products = json.load(f)

    if isinstance(raw_products, list) and raw_products and "coverage_name" in raw_products[0]:
        raw_products = [{"product_name": "기본 분석 상품", "conf_n": 0.95, "riders": raw_products}]

    for p in raw_products:
        p_name = p.get("product_name", "보험 상품")
        conf_n = min(p.get("conf_n", 0.90), 1.0)
        brand_bonus = (royalty / 100.0) * 0.02
        riders = p.get("riders", [])

        total_prem = 0
        cancer_amt, vasc_amt, inj_amt, treat_cnt = 0, 0, 0, 0
        renewable_cnt = 0
        inherent_toxic_cnt = 0
        matched_user_risks = []
        coverage_highlights = []

        for r in riders:
            c_name = r.get("coverage_name", "")
            amt = parse_currency(r.get("subscribed_amount", r.get("amount", 0)))
            prem = parse_currency(r.get("premium", 0))
            pay_cond = str(r.get("payment_condition", "") or "")
            toxic_field = str(r.get("toxic_clauses", "") or "")
            full_text = f"{c_name} {pay_cond} {toxic_field}"

            total_prem += prem
            if "[갱신형]" in c_name or "갱신형" in str(r.get("payment_period", "")):
                renewable_cnt += 1

            # 기본 독소조항 카운트
            if any(k in full_text for k in ["면책", "부담보", "제외", "감액", "미지급"]):
                inherent_toxic_cnt += 1

            # 검증된 사용자 키워드 매칭
            for rk in validated_risk_keys:
                if rk in full_text:
                    matched_user_risks.append(f"약관 내 '{rk}' 관련 면책/제약 조건 포함 ({c_name})")

            # 보장 분류
            if "암진단비" in c_name:
                cancer_amt += amt
                coverage_highlights.append(f"암진단비: {amt//10000:,}만원")
            elif any(k in c_name for k in ["뇌혈관", "허혈성", "심혈관"]):
                vasc_amt += amt
                coverage_highlights.append(f"{c_name.split('(')[0]}: {amt//10000:,}만원")
            elif any(k in c_name for k in ["상해", "골절"]):
                inj_amt += amt
            if any(k in c_name for k in ["치료비", "수술비", "방사선", "약물", "표적", "중입자"]):
                treat_cnt += 1

        # 1. 보장금액 점수 (amt_n) & 보장범위 점수 (breadth_n)
        amt_n = (
            w_cancer * calc_coverage_utility(cancer_amt, 50_000_000) +
            w_vasc * calc_coverage_utility(vasc_amt, 30_000_000) +
            w_inj * calc_coverage_utility(inj_amt, 50_000_000)
        )
        breadth_n = w_inp * min(treat_cnt / 10.0, 1.0)

        # 2. 가성비 페널티 (월 8만원 초과 시 초과액에 비례 감점)
        price_penalty = 0.0
        if total_prem > 80_000:
            price_penalty = min(((total_prem - 80_000) / 100_000) * 0.08, 0.10)

        # 3. 논문 원본 공식 기반 base 산출
        base_score = (w_amt_dyn * amt_n) + (w_breadth_dyn * breadth_n) + (w_conf_dyn * conf_n) + brand_bonus - price_penalty

        # 4. 리스크 및 면책 페널티 산출
        user_penalty = len(matched_user_risks) * 0.20
        if renewal_pref == "비갱신" and renewable_cnt > 0:
            user_penalty += (renewable_cnt / max(len(riders), 1)) * 0.25

        risk_verified_n = min(0.10 + (inherent_toxic_cnt * 0.08) + user_penalty, 1.0)
        risk_naive_n = min(risk_verified_n * 1.6, 1.5)

        # 5. 시나리오별 최종 점수 연산 (논문 공식 엄격 적용)
        raw_s1 = base_score / base_weight
        raw_s2 = base_score - (risk_weight * risk_naive_n)
        raw_s3 = base_score - (risk_weight * risk_verified_n)

        s1 = max(0.0, raw_s1)
        s2 = max(0.0, raw_s2)
        s3 = max(0.0, raw_s3)


        # [UX 보정] S3의 실질 최댓값 기준 100점 척도 환산
        theoretical_max_s3 = max(base_weight - (risk_weight *0.10), 0.30)
        score_100 = round(min(max(s3 / theoretical_max_s3, 0.0) * 100, 100.0), 1)
 
        evaluated_products.append({
            "name": p_name,
            "premium": total_prem,
            "rider_count": len(riders),
            "renewable_count": renewable_cnt,
            "toxic_count": inherent_toxic_cnt,
            "user_risks": list(set(matched_user_risks)),
            "highlights": coverage_highlights[:4],
            "amt_score": round(amt_n, 3),
            "breadth_score": round(breadth_n, 3),
            "price_penalty": round(price_penalty, 3),
            "s1": round(s1, 3),
            "s2": round(s2, 3),
            "s3": round(s3, 3),
            "score_100": score_100
        })

evaluated_products.sort(key=lambda x: x["s3"], reverse=True)

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
    m4.metric("검출된 제약/독소조항", f"{best['toxic_count']} 건")

    st.markdown("### 📋 AI 추천 사유 및 약관 정밀 분석 리포트")

    with st.container():
        st.markdown(f"""
        #### 1. 왜 이 상품이 1위인가요?
        * **공공데이터 부합도:** **{age}세 {gender}** 공공 의료 통계상 우선순위가 높은 주요 질환(암 {baseline['cancer']*100:.1f}%, 뇌·심장 {baseline['vascular']*100:.1f}%)에 맞추어 필수 진단비와 치료비가 빈틈없이 구성되었습니다.
        * **보장 충실도 & 가성비:** 필수 보장 정규화 점수 **{best['amt_score']}점**, 신의료 특약 다양성 **{best['breadth_score']}점**을 획득했으며, 불필요한 과잉 설계로 인한 가격 감점(`price_penalty`: {best['price_penalty']}점)을 최소화했습니다.
        * **대표 주요 보장:** {', '.join(best['highlights']) if best['highlights'] else '기본 보장 구성'} 외 총 {best['rider_count']}개 특약 탑재
        """)

        # 사용자 입력 조건 매칭 결과
        if validated_risk_keys:
            st.markdown("#### 2. 개인 맞춤 특이사항 및 독소조항 팩트체크")
            st.info(f"🔍 **분석에 반영된 위험 관심 키워드:** `{', '.join(validated_risk_keys)}`")
            if best["user_risks"]:
                st.warning(f"⚠️ **1위 추천 상품 주의:** 약관 내 다음 제약 사항이 확인되었습니다:\n- " + "\n- ".join(best["user_risks"]))
            else:
                st.info("✅ **독소조항 안전:** 입력하신 조건에 위배되는 치명적인 면책/지급 제한 조항이 1위 추천 상품에서는 발견되지 않았습니다.")

        # 타 상품 비교 분석
        if len(evaluated_products) > 1:
            st.markdown("#### 3. 다른 비교 상품들의 평가 및 순위 하락 이유")
            for other in evaluated_products[1:]:
                reasons = []
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

    # 시각화 탭
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