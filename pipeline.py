import json
import re
import math
import numpy as np
import ollama
import streamlit as st

OLLAMA_MODEL_NAME = "qwen2.5:7b"

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

# --- 2. 통화 파싱 및 효용 함수(과도하게 큰 금액에 대한 보장 금액을 로그함수를 통해 점수 상승폭을 억제)---
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

# --- 5. 통합 평가 파이프라인 (5개 영역 반영) => 아제나리스크와 베이지안 추론 이용 ---
def run_evaluation_pipeline(raw_products: list, user_prefs: dict, user_notes: str, selected_tags: list):
    w_final = user_prefs["w_final"]
    claim_rate_val = user_prefs["claim_rate_val"]
    renewal_pref = user_prefs["renewal_pref"]
    D = user_prefs.get("D", 0.0)

    # 증거에 따른 위험 민감도 동적 갱신
    sensitivity_multiplier = 1.0 + (D * 0.10)

    # 사전 위험 가중치 (목표 지급률 + 편차 증거 반영)
    risk_weight = (0.20 + (claim_rate_val / 100.0) * 0.40) * min(sensitivity_multiplier, 1.3)
    risk_weight = min(risk_weight, 0.70)
    base_weight = 1.0 - risk_weight

    w_amt_dyn = base_weight * 0.55
    w_breadth_dyn = base_weight * 0.30
    w_conf_dyn = base_weight * 0.15

    validated_risk_keys = extract_valid_risk_intents(user_notes, selected_tags)
    evaluated_products = []

    if isinstance(raw_products, list) and raw_products and "coverage_name" in raw_products[0]:
        raw_products = [{"product_name": "기본 분석 상품", "conf_n": 0.95, "riders": raw_products}]

    for p in raw_products:
        p_name = p.get("product_name", "보험 상품")
        conf_n = min(p.get("conf_n", 0.90), 1.0)
        riders = p.get("riders", [])
        total_prem = 0
        cancer_amt, brain_amt, heart_amt, inj_amt, inp_amt = 0, 0, 0, 0, 0
        treat_cnt = 0
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

            # 5대 기준 특약 가입금액 파싱
            if "암진단비" in c_name:
                cancer_amt += amt
                coverage_highlights.append(f"암진단비: {amt//10000:,}만원")
            elif any(k in c_name for k in ["뇌혈관", "뇌졸중", "뇌출혈", "뇌경색"]):
                brain_amt += amt
                coverage_highlights.append(f"{c_name.split('(')[0]}: {amt//10000:,}만원")
            elif any(k in c_name for k in ["허혈성", "심혈관", "심근경색", "부정맥", "심장"]):
                heart_amt += amt
                coverage_highlights.append(f"{c_name.split('(')[0]}: {amt//10000:,}만원")
            elif any(k in c_name for k in ["상해", "골절"]):
                inj_amt += amt
            elif any(k in c_name for k in ["입원", "수술", "일당", "간병"]):
                inp_amt += amt

            if any(k in c_name for k in ["치료비", "수술비", "방사선", "약물", "표적", "중입자"]):
                treat_cnt += 1

        llm_toxic_details = []
        if user_notes.strip() and raw_clause_texts:
            llm_toxic_details = extract_toxic_clauses_with_llm(user_notes, tuple(raw_clause_texts[:35]))

        # 5대 기준 w_final 효용 합산 (단위: 원) => Weighted Scoring 적용
        amt_n = (
            w_final["암"] * calc_coverage_utility(cancer_amt, 50_000_000) +
            w_final["뇌혈관"] * calc_coverage_utility(brain_amt, 30_000_000) +
            w_final["심장"] * calc_coverage_utility(heart_amt, 30_000_000) +
            w_final["상해"] * calc_coverage_utility(inj_amt, 50_000_000) +
            w_final["입원/수술"] * calc_coverage_utility(inp_amt, 20_000_000)
        )
        breadth_n = min(treat_cnt / 10.0, 1.0)

        price_penalty = 0.0
        if total_prem > 80_000:
            price_penalty = min(((total_prem - 80_000) / 100_000) * 0.08, 0.10)

        # 최상위 평가 기준 간의 가중 합산
        base_score = (w_amt_dyn * amt_n) + (w_breadth_dyn * breadth_n) + (w_conf_dyn * conf_n) - price_penalty

        llm_penalty = len(llm_toxic_details) * 0.20 * sensitivity_multiplier
        user_penalty = (len(matched_user_risks) * 0.20 * sensitivity_multiplier) + llm_penalty
        if renewal_pref == "비갱신" and renewable_cnt > 0:
            user_penalty += (renewable_cnt / max(len(riders), 1)) * 0.25

        risk_verified_n = min(0.10 + (inherent_toxic_cnt * 0.08) + user_penalty, 1.0)
        risk_naive_n = min(risk_verified_n * 1.6, 1.5)

        raw_s1 = base_score / base_weight
        raw_s2 = base_score - (risk_weight * risk_naive_n)

        # 최종 점수 도출
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