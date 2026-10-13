import json
import re
import math
import numpy as np
import ollama
import streamlit as st

OLLAMA_MODEL_NAME = "qwen2.5:7b"

# -------------------------------------------------------------------------
# [헬퍼 함수] 보험사명 자동 감지 함수
# -------------------------------------------------------------------------
def extract_company_name(p: dict) -> str:
    raw_company = p.get("company")
    if raw_company and raw_company not in ["기타", "DB손해보험"]:
        return raw_company

    p_name = p.get("product_name", "") or p.get("name", "")
    known_companies = [
        "DB손해보험", "현대해상", "한화손해보험", "한화생명",
        "삼성화재", "삼성생명", "KB손해보험", "메리츠화재",
        "흥국화재", "교보생명", "신한라이프", "NH농협손해"
    ]
    for comp in known_companies:
        short_name = comp[:2]
        if short_name in p_name or comp in p_name:
            return comp

    return raw_company if raw_company else "기타손해보험"

# --- 1. AHP 쌍대비교 및 편차 보정 알고리즘 ---
def calculate_ahp_weights(matrix: np.ndarray, criteria: list) -> dict:
    """AHP 쌍대비교 행렬의 기하평균법 기반 가중치(w_user) 산출"""
    n = len(criteria)
    geo_means = np.prod(matrix, axis=1) ** (1.0 / n)
    weights = geo_means / np.sum(geo_means)
    return {criteria[i]: round(float(weights[i]), 4) for i in range(n)}

def apply_delta_correction(w_base: dict, w_user: dict, tau: float = 0.3):
    """
    3단계 편차 보정 수식
    Delta_i = w_user_i - w_base_i
    D = sum(|Delta_i|)
    alpha = min(1, tau / D)
    w_final_i = w_base_i + alpha * Delta_i
    """
    keys = list(w_base.keys())
    deltas = {k: w_user[k] - w_base[k] for k in keys}
    D = sum(abs(v) for v in deltas.values())
    
    alpha = min(1.0, tau / D) if D > 0 else 1.0
    w_final = {k: round(w_base[k] + alpha * deltas[k], 4) for k in keys}
    
    total_w = sum(w_final.values())
    if total_w > 0:
        w_final = {k: round(v / total_w, 4) for k, v in w_final.items()}
        
    return w_final, deltas, alpha, D

# --- 2. 통화 파싱 및 효용 함수 ---
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
    if digits and ("만" not in str(val) and "억" not in str(val)):
        total += int(digits[0])
    return total

def calc_coverage_utility(amt, baseline=50_000_000):
    if amt <= 0: return 0.0
    if amt <= baseline: return amt / baseline
    return min(1.0 + 0.25 * math.log10(amt / baseline + 1), 1.25)

# --- 3. 정규식 부정어 판별 엔진 ---
def extract_valid_risk_intents(user_notes: str, selected_tags: list) -> list:
    extracted = set()
    tag_mapping = {
        "이륜차/오토바이 운전": "이륜차",
        "전이암/유사암 보장 필수": "전이암",
        "비갱신형 상품 선호": "갱신",
        "감액/면책기간 최소화 희망": "감액"
    }
    for tag in selected_tags:
        if tag in tag_mapping:
            extracted.add(tag_mapping[tag])

    if user_notes:
        DOMAIN_RISK_MAP = {
            "이륜차": ["이륜차", "오토바이", "바이크", "스쿠터", "원동기"],
            "전이암": ["전이암", "재발암", "유사암", "소액암"],
            "뇌혈관": ["뇌혈관", "뇌졸중", "뇌출혈", "뇌경색"],
            "심혈관": ["허혈성", "심근경색", "부정맥", "협심증", "심장"],
            "감액": ["감액", "면책기간", "대기기간"]
        }
        NEGATION_PATTERN = r"(안함|안 함|않음|않아|안탐|안 타|비운전|해당없음|없음|상관없음)"

        for standard_key, synonyms in DOMAIN_RISK_MAP.items():
            for syn in synonyms:
                if syn in user_notes:
                    pattern_after = rf"{syn}.{{0,8}}{NEGATION_PATTERN}"
                    pattern_before = rf"{NEGATION_PATTERN}.{{0,8}}{syn}"

                    is_negated = bool(re.search(pattern_after, user_notes) or re.search(pattern_before, user_notes))
                    if not is_negated or ("제외" in user_notes and "피하" in user_notes):
                        extracted.add(standard_key)
                        break
    return list(extracted)

# --- 4. Ollama LLM 독소조항 분석 엔진 ---
@st.cache_data(show_spinner="LLM이 특약 약관의 독소조항을 정밀 분석 중입니다...", ttl=60)
def extract_toxic_clauses_with_llm(user_input: str, clauses_tuple: tuple) -> list:
    if not user_input.strip() or not clauses_tuple:
        return []

    clauses_text = "\n".join([f"[{idx+1}] {text}" for idx, text in enumerate(clauses_tuple)])

    prompt = f"""당신은 보험 약관 분석 및 소비자 권익 보호 전문가입니다.
사용자의 요구사항/기피조건을 확인하고, 제공된 특약 약관 리스트 중 사용자에게 불리하거나 면책(지급 거절/제한) 사유가 되는 '독소조항'을 식별하세요.
위험/독소조항이 없다면 빈 리스트 [] 를 반환하세요.

[사용자 기피/특이사항]:
"{user_input}"

[분석 대상 특약 약관]:
{clauses_text}

반드시 다른 설명 없이 아래 JSON 리스트 포맷으로만 응답하세요:
[
  {{
    "target_name": "해당 특약명",
    "issue_summary": "독소조항 핵심 요약 (1~2문장)",
    "severity": "HIGH 또는 MEDIUM"
  }}
]
"""
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}],
            format='json',
            options={'temperature': 0.1, 'num_predict': 512}
        )
        raw_content = response['message']['content'].strip()

        json_match = re.search(r'\[\s*\{.*?\}\s*\]', raw_content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))

        single_match = re.search(r'\{.*?\}', raw_content, re.DOTALL)
        if single_match:
            return [json.loads(single_match.group(0))]

        if "[]" in raw_content:
            return []

        return json.loads(raw_content)

    except Exception as e:
        print(f"❌ [Ollama 호출 에러]: {e}")
        return []

# --- 5. 통합 평가 파이프라인 ---
def run_evaluation_pipeline(raw_products: list, user_prefs: dict, user_notes: str, selected_tags: list):
    w_final = user_prefs["w_final"]
    claim_rate_val = user_prefs["claim_rate_val"]
    renewal_pref = user_prefs["renewal_pref"]
    D = user_prefs.get("D", 0.0)

    # =========== 편차(D)에 따른 리스크 민감도 동적 가중 =============
    sensitivity_multiplier = 1.0 + (D * 0.10)
    risk_weight = (0.20 + (claim_rate_val / 100.0) * 0.40) * min(sensitivity_multiplier, 1.3)
    risk_weight = min(risk_weight, 0.65)
    base_weight = max(1.0 - risk_weight, 0.35)

    w_amt_dyn = base_weight * 0.75             # 이 코드 포함 아래 3줄의 비율 합이 1.0이 되도록 조정해야함 
    w_breadth_dyn = base_weight * 0.15         # amt=coverage, breadth=coverage breadth/diversity, conf=confidence in product info
    w_conf_dyn = base_weight * 0.10
    # =============================================================================

    validated_risk_keys = extract_valid_risk_intents(user_notes, selected_tags)
    evaluated_products = []

    if isinstance(raw_products, list) and raw_products and "coverage_name" in raw_products[0]:
        raw_products = [{"product_name": "기본 분석 상품", "conf_n": 0.95, "riders": raw_products}]

    # Min-Max 사전 집계
    all_money_values = []
    all_coverage_years = []

    for p in raw_products:
        payment_years = int(p.get("payment_years", 20) or 20)
        entry_age = int(p.get("entry_age", 20) or 20)
        maturity_age = int(p.get("maturity_age", 90) or 90)
        c_years = max(maturity_age - entry_age, 1)
        all_coverage_years.append(c_years)

        for r in p.get("riders", []):
            amt = parse_currency(r.get("subscribed_amount", r.get("amount", 0)))
            prem = parse_currency(r.get("premium", 0))
            cov_cost = prem * 12 * payment_years
            if cov_cost > 0 and amt > 0:
                all_money_values.append(amt / cov_cost)

    min_money = min(all_money_values) if all_money_values else 0.0
    max_money = max(all_money_values) if all_money_values else 1.0
    min_years = min(all_coverage_years) if all_coverage_years else 10
    max_years = max(all_coverage_years) if all_coverage_years else 80

    def min_max_norm(val, v_min, v_max):
        if v_max <= v_min:
            return 0.5
        return (val - v_min) / (v_max - v_min)

    c_score_avg = lambda lst: (sum(lst) / len(lst)) if lst else 0.0

    # =========================================================================
    # [PDF 가이드 2단계] 개별 상품 평가 루프
    # =========================================================================
    for p in raw_products:
        p_name = p.get("product_name", "보험 상품")
        conf_n = min(p.get("conf_n", 0.90), 1.0)
        riders = p.get("riders", [])
        total_prem = 0
        payment_years = int(p.get("payment_years", 20) or 20)
        entry_age = int(p.get("entry_age", 20) or 20)
        maturity_age = int(p.get("maturity_age", 90) or 90)
        c_years = max(maturity_age - entry_age, 1)

        period_score = min_max_norm(c_years, min_years, max_years)

        cancer_scores, brain_scores, heart_scores, inj_scores, inp_scores = [], [], [], [], []
        treat_cnt = 0
        renewable_cnt = 0
        inherent_toxic_cnt = 0
        matched_user_risks = []
        coverage_highlights = []
        raw_clause_texts = []
        llm_toxic_details = []

        for r in riders:
            c_name = r.get("coverage_name", "")
            amt = parse_currency(r.get("subscribed_amount", r.get("amount", 0)))
            prem = parse_currency(r.get("premium", 0))
            pay_cond = str(r.get("payment_condition", "") or "")
            toxic_field = str(r.get("toxic_clauses", "") or "")
            pay_period = str(r.get("payment_period", "") or "")
            full_text = f"{c_name} {pay_cond} {toxic_field}".strip()

            if full_text:
                raw_clause_texts.append(full_text)

            total_prem += prem

            cov_cost = prem * 12 * payment_years
            money_val = (amt / cov_cost) if cov_cost > 0 else 0.0
            money_score = min_max_norm(money_val, min_money, max_money)
            coverage_score = 0.6 * money_score + 0.4 * period_score

            if "갱신" in c_name or "갱신" in pay_period:
                renewable_cnt += 1

            if any(k in full_text for k in ["면책", "부담보", "제외", "감액", "미지급"]):
                inherent_toxic_cnt += 1

            for rk in validated_risk_keys:
                if rk in full_text:
                    matched_user_risks.append(f"약관 내 '{rk}' 관련 면책/제약 조건 포함 ({c_name})")

            if "암진단비" in c_name:
                cancer_scores.append(coverage_score)
                coverage_highlights.append(f"암진단비: {amt//10000:,}만원")
            elif any(k in c_name for k in ["뇌혈관", "뇌졸중", "뇌출혈", "뇌경색"]):
                brain_scores.append(coverage_score)
                coverage_highlights.append(f"{c_name.split('(')[0]}: {amt//10000:,}만원")
            elif any(k in c_name for k in ["허혈성", "심혈관", "심근경색", "부정맥", "심장"]):
                heart_scores.append(coverage_score)
                coverage_highlights.append(f"{c_name.split('(')[0]}: {amt//10000:,}만원")
            elif any(k in c_name for k in ["상해", "골절"]):
                inj_scores.append(coverage_score)
            elif any(k in c_name for k in ["입원", "수술", "일당", "간병"]):
                inp_scores.append(coverage_score)

            if any(k in c_name for k in ["치료비", "수술비", "방사선", "약물", "표적", "중입자"]):
                treat_cnt += 1

        # [PDF 가이드 결합] 5대 기준별 coverage_score 평균과 AHP w_final 가중합 산출
        amt_n = (
            w_final.get("암", 0.2) * c_score_avg(cancer_scores) +
            w_final.get("뇌혈관", 0.2) * c_score_avg(brain_scores) +
            w_final.get("심장", 0.2) * c_score_avg(heart_scores) +
            w_final.get("상해", 0.2) * c_score_avg(inj_scores) +
            w_final.get("입원/수술", 0.2) * c_score_avg(inp_scores)
        )
        amt_n = max(amt_n, 0.40)
        breadth_n = min(treat_cnt / 10.0, 1.0)

        price_penalty = 0.0
        if total_prem > 80_000:
            price_penalty = min(((total_prem - 80_000) / 100_000) * 0.08, 0.10)

        base_score = (w_amt_dyn * amt_n) + (w_breadth_dyn * breadth_n) + (w_conf_dyn * conf_n) - price_penalty

        llm_penalty = len(llm_toxic_details) * 0.20 * sensitivity_multiplier
        user_penalty = (len(matched_user_risks) * 0.20 * sensitivity_multiplier) + llm_penalty

        # --- [개선] 갱신형/비갱신형/미지정 선호도 양방향 반영 로직 ---
        total_riders = max(len(riders), 1)
        renewable_ratio = renewable_cnt / total_riders

        if renewal_pref == "비갱신":
            if renewable_cnt > 0:
                user_penalty += renewable_ratio * 0.30
        elif renewal_pref == "갱신":
            non_renewable_ratio = 1.0 - renewable_ratio
            if non_renewable_ratio > 0.4:
                user_penalty += non_renewable_ratio * 0.25

        risk_verified_n = min(0.05 + (inherent_toxic_cnt * 0.03) + user_penalty, 1.0)
        risk_naive_n = min(risk_verified_n * 1.4, 1.2)

        # [1] 지수 감쇄 적용
        retention_s3 = math.exp(-1.3 * risk_weight * risk_verified_n)
        retention_s2 = math.exp(-1.3 * risk_weight * risk_naive_n)

        raw_s1 = base_score / max(base_weight, 0.1)
        raw_s2 = base_score * retention_s2
        raw_s3 = base_score * retention_s3

        s1 = max(0.0, raw_s1)
        s2 = max(0.0, raw_s2)
        s3 = max(0.0, raw_s3)

# ====================== 종합 점수 조절을 위한 수치  ========================
        # [2] 100점 환산 기준값 (지수 모델 매핑)
        theoretical_max_s3 = max(base_weight * 0.65, 0.22)  # 앞 숫자가 높을수록 100점 환산 점수 기준이 올라감, 뒤 숫자는 오버슈팅 방지 안전장치
        calculated_100 = (s3 / theoretical_max_s3) * 100.0
        score_100 = round(min(max(calculated_100, 10.0), 100.0), 1) # 앞 숫자는 패널티 맞은 점수 최소 보장 마지노선, 뒤 숫자는 100점 넘지 못하게 하는 친구
# ==============================================================================

        # [3] 5대 영역별 가중 만족도 점수 산출
        weight_keys = {
            "암": "암",
            "뇌 질환": "뇌혈관",
            "심장 질환": "심장",
            "상해": "상해",
            "입원/수술": "입원/수술",
        }


# ================= 항목별 결과 상태바 점수 조절 수치 =================
        category_raw = {
            "암": c_score_avg(cancer_scores) if cancer_scores else 0.35,            # 특약 매칭이 안되었을 경우 최소 점수 0.35로 설정 (0~1 범위)
            "뇌 질환": c_score_avg(brain_scores) if brain_scores else 0.35,
            "심장 질환": c_score_avg(heart_scores) if heart_scores else 0.35,
            "상해": c_score_avg(inj_scores) if inj_scores else 0.35,
            "입원/수술": c_score_avg(inp_scores) if inp_scores else 0.35,
        }
        category_scores = {}
        for cat_name, raw_val in category_raw.items():
            w_k = weight_keys[cat_name]
            w_val = w_final.get(w_k, 0.20)
            w_ratio = (w_val / 0.20) ** 0.7      # 앞 숫자는 슬라이더 항목별 균등 기본 비중 / 뒤 숫자는 슬라이더 반응 민감도 지수
            scaled_score = 52.0 + (raw_val * 42.0 * w_ratio)  # 앞 숫자 기본 점수, 뒤 숫자는 "최대점수 = (기본점수 + @)" 에서  @값
            category_scores[cat_name] = round(min(max(scaled_score, 20.0), 96.0), 1) # 앞 숫자는 최소 점수, 뒤 숫자는 최대 점수, 마지막은 반올림 자리수
# ==============================================================================

        evaluated_products.append({
            "name": p_name,
            "company": extract_company_name(p),
            "premium": total_prem,
            "rider_count": len(riders),
            "renewable_count": renewable_cnt,
            "toxic_count": inherent_toxic_cnt,
            "user_risks": list(set(matched_user_risks)),
            "llm_toxics": [],
            "highlights": coverage_highlights,
            "amt_score": round(amt_n, 3),
            "breadth_score": round(breadth_n, 3),
            "price_penalty": round(price_penalty, 3),
            "s1": round(s1, 3),
            "s2": round(s2, 3),
            "s3": round(s3, 3),
            "score_100": score_100,
            "category_scores": category_scores,
            "raw_clauses": raw_clause_texts
        })

    # =========================================================================
    # [for 루프 바깥] 상위 3개 대상 빠른 Ollama 정밀 검증 및 S3 최종 감쇄
    # =========================================================================
    evaluated_products.sort(key=lambda x: x["s3"], reverse=True)
    top_candidates = evaluated_products[:3]
    other_candidates = evaluated_products[3:]

    if user_notes.strip():
        for item in top_candidates:
            raw_clauses = item.get("raw_clauses", [])
            if raw_clauses:
                limited_clauses = tuple(raw_clauses[:5])
                detected_toxics = extract_toxic_clauses_with_llm(user_notes, limited_clauses)
                item["llm_toxics"] = detected_toxics if isinstance(detected_toxics, list) else []

                # LLM 독소조항 검출 시 지수 감쇄 공식 적용
                if item["llm_toxics"]:
                    risk_factor = len(item["llm_toxics"]) * 0.35
                    item["s3"] = round(item["s3"] * math.exp(-0.25 * risk_factor), 3)
                    theoretical_max_s3 = max(base_weight * 0.95, 0.20)
                    item["score_100"] = round(min(max((item["s3"] / theoretical_max_s3) * 100.0, 3.0), 100.0), 1)

    sorted_products = sorted(top_candidates + other_candidates, key=lambda x: x["s3"], reverse=True)
    return sorted_products, validated_risk_keys