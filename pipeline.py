import json
import re
import math
import ollama
import streamlit as st

OLLAMA_MODEL_NAME = "qwen2.5:7b"

# --- 1. 통화 파싱 및 효용 함수 ---
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

def calc_coverage_utility(amt, baseline=50_000_000):
    if amt <= 0: return 0.0
    if amt <= baseline: return amt / baseline
    return min(1.0 + 0.25 * math.log10(amt / baseline + 1), 1.25)

# --- 2. 정규식 부정어 판별 엔진 ---
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
            "심혈관": ["허혈성", "심근경색", "부정맥", "협심증"],
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

# --- 3. Ollama LLM 독소조항 분석 엔진 ---
@st.cache_data(show_spinner="LLM이 특약 약관의 독소조항을 정밀 분석 중입니다...")
def extract_toxic_clauses_with_llm(user_input: str, clauses_tuple: tuple) -> list:
    if not user_input.strip() or not clauses_tuple:
        return []

    clauses_text = "\n".join([f"[{idx+1}] {text}" for idx, text in enumerate(clauses_tuple)])

    prompt = f"""당신은 보험 약관 분석 및 소비자 권익 보호 전문가입니다.
사용자의 요구사항/기피조건을 확인하고, 제공된 특약 약관 리스트 중 사용자에게 불리하거나 면책(지급 거절/제한) 사유가 되는 '독소조항'을 식별하세요.

[사용자 기피/특이사항]:
"{user_input}"

[분석 대상 특약 약관]:
{clauses_text}

반드시 아래 JSON 포맷으로만 응답하세요.
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
            options={'temperature': 0.1, 'num_predict': 512}
        )
        content = response['message']['content'].strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content.strip())
    except Exception:
        return []

# --- 4. 통합 평가 파이프라인 (메인 진입점) ---
def run_evaluation_pipeline(raw_products: list, user_prefs: dict, user_notes: str, selected_tags: list):
    cancer_val = user_prefs["cancer_val"]
    vascular_val = user_prefs["vascular_val"]
    injury_val = user_prefs["injury_val"]
    inpatient_val = user_prefs["inpatient_val"]
    claim_rate_val = user_prefs["claim_rate_val"]
    royalty = user_prefs["royalty"]
    renewal_pref = user_prefs["renewal_pref"]

    raw_w = [cancer_val, vascular_val, injury_val, inpatient_val]
    sum_w = sum(raw_w) if sum(raw_w) > 0 else 1.0
    w_cancer, w_vasc, w_inj, w_inp = [v / sum_w for v in raw_w]

    risk_weight = 0.20 + (claim_rate_val / 100.0) * 0.40
    base_weight = 1.0 - risk_weight

    w_amt_dyn = base_weight * (0.30 / 0.60)
    w_breadth_dyn = base_weight * (0.20 / 0.60)
    w_conf_dyn = base_weight * (0.10 / 0.60)

    validated_risk_keys = extract_valid_risk_intents(user_notes, selected_tags)
    evaluated_products = []

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
        raw_clause_texts = []

        for r in riders:
            c_name = r.get("coverage_name", "")
            amt = parse_currency(r.get("subscribed_amount", r.get("amount", 0)))
            prem = parse_currency(r.get("premium", 0))
            pay_cond = str(r.get("payment_condition", "") or "")
            toxic_field = str(r.get("toxic_clauses", "") or "")
            full_text = f"{c_name} {pay_cond} {toxic_field}".strip()
            if full_text:
                raw_clause_texts.append(full_text)

            total_prem += prem
            if "[갱신형]" in c_name or "갱신형" in str(r.get("payment_period", "")):
                renewable_cnt += 1

            if any(k in full_text for k in ["면책", "부담보", "제외", "감액", "미지급"]):
                inherent_toxic_cnt += 1

            for rk in validated_risk_keys:
                if rk in full_text:
                    matched_user_risks.append(f"약관 내 '{rk}' 관련 면책/제약 조건 포함 ({c_name})")

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

        # LLM 독소조항 감지
        llm_toxic_details = []
        if user_notes.strip() and raw_clause_texts:
            llm_toxic_details = extract_toxic_clauses_with_llm(user_notes, tuple(raw_clause_texts[:35]))

        amt_n = (
            w_cancer * calc_coverage_utility(cancer_amt, 50_000_000) +
            w_vasc * calc_coverage_utility(vasc_amt, 30_000_000) +
            w_inj * calc_coverage_utility(inj_amt, 50_000_000)
        )
        breadth_n = w_inp * min(treat_cnt / 10.0, 1.0)

        price_penalty = 0.0
        if total_prem > 80_000:
            price_penalty = min(((total_prem - 80_000) / 100_000) * 0.08, 0.10)

        base_score = (w_amt_dyn * amt_n) + (w_breadth_dyn * breadth_n) + (w_conf_dyn * conf_n) + brand_bonus - price_penalty

        llm_penalty = len(llm_toxic_details) * 0.20
        user_penalty = (len(matched_user_risks) * 0.20) + llm_penalty
        if renewal_pref == "비갱신" and renewable_cnt > 0:
            user_penalty += (renewable_cnt / max(len(riders), 1)) * 0.25

        risk_verified_n = min(0.10 + (inherent_toxic_cnt * 0.08) + user_penalty, 1.0)
        risk_naive_n = min(risk_verified_n * 1.6, 1.5)

        raw_s1 = base_score / base_weight
        raw_s2 = base_score - (risk_weight * risk_naive_n)
        raw_s3 = base_score - (risk_weight * risk_verified_n)

        s1 = max(0.0, raw_s1)
        s2 = max(0.0, raw_s2)
        s3 = max(0.0, raw_s3)

        theoretical_max_s3 = max(base_weight - (risk_weight * 0.10), 0.30)
        score_100 = round(min(max(s3 / theoretical_max_s3, 0.0) * 100, 100.0), 1)

        evaluated_products.append({
            "name": p_name,
            "premium": total_prem,
            "rider_count": len(riders),
            "renewable_count": renewable_cnt,
            "toxic_count": inherent_toxic_cnt,
            "user_risks": list(set(matched_user_risks)),
            "llm_toxics": llm_toxic_details,
            "highlights": coverage_highlights[:4],
            "amt_score": round(amt_n, 3),
            "breadth_score": round(breadth_n, 3),
            "price_penalty": round(price_penalty, 3),
            "s1": round(s1, 3),
            "s2": round(s2, 3),
            "s3": round(s3, 3),
            "score_100": score_100
        })

    sorted_products = sorted(evaluated_products, key=lambda x: x["s3"], reverse=True)
    return sorted_products, validated_risk_keys