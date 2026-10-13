import json
import os
import re
import unicodedata
import hashlib

# 스크립트가 위치한 Microstone_2026 루트 디렉터리 기준
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def nfc(text: str) -> str:
    """Mac(NFD)과 Windows(NFC) 한글 자모 분리 방지 정규화"""
    if not isinstance(text, str):
        return text
    return unicodedata.normalize("NFC", text)

def parse_int_years(val, default_val=20):
    if not val:
        return default_val
    digits = re.findall(r"\d+", str(val))
    return int(digits[0]) if digits else default_val

def parse_age(age_val, age_raw):
    if age_val is not None:
        try:
            return int(age_val)
        except:
            pass
    if age_raw:
        digits = re.findall(r"\d+", str(age_raw))
        if digits:
            return int(digits[0])
    return 30

def extract_amount(sub_amt):
    """가입금액 단위(억, 만, 원) 파싱"""
    if not sub_amt:
        return 0
    if isinstance(sub_amt, dict):
        if sub_amt.get("annual_limit"):
            return int(sub_amt["annual_limit"])
        if sub_amt.get("per_time"):
            return int(sub_amt["per_time"])
        raw = sub_amt.get("raw", "")
    else:
        raw = str(sub_amt)

    clean_str = str(raw).replace(",", "").replace(" ", "").strip()
    total = 0
    if "억" in clean_str:
        p = clean_str.split("억")
        d = re.findall(r"\d+", p[0])
        if d:
            total += int(d[0]) * 100_000_000
        clean_str = p[1] if len(p) > 1 else ""
    if "만" in clean_str:
        p = clean_str.split("만")
        d = re.findall(r"\d+", p[0])
        if d:
            total += int(d[0]) * 10_000
        clean_str = p[1] if len(p) > 1 else ""
    d = re.findall(r"\d+", clean_str)
    if d and ("만" not in str(raw) and "억" not in str(raw)):
        total += int(d[0])
    return total

def load_all_toxic_clauses():
    """3_약관/독소조항 폴더 내 JSON 파싱 및 캐싱"""
    toxic_dir = os.path.join(BASE_DIR, "3_약관", "독소조항")
    toxic_cache = {}

    if not os.path.exists(toxic_dir):
        print(f"⚠️ 독소조항 폴더가 없습니다: {toxic_dir}")
        return toxic_cache

    for root, _, files in os.walk(toxic_dir):
        for f in files:
            f_norm = nfc(f)
            if f_norm.endswith("_toxic.json"):
                f_path = os.path.join(root, f)
                try:
                    with open(f_path, "r", encoding="utf-8") as jf:
                        tdata = json.load(jf)

                    snippets = []
                    raw_chunks = []
                    for art in tdata.get("toxic_clauses", []):
                        art_title = nfc(art.get("article_title", ""))
                        for tc in art.get("toxic_clauses", []):
                            s = nfc(tc.get("clause_summary", ""))
                            r = nfc(tc.get("reason", ""))
                            sev = nfc(tc.get("severity", "중간"))
                            snippets.append(f"[{sev}] {s}: {r}")
                            quote = nfc(tc.get("source_quote", ""))
                            if quote:
                                raw_chunks.append(f"[{art_title}] {quote}")

                    key = nfc(f_norm.replace("_toxic.json", "").replace(".pdf", ""))
                    toxic_cache[key] = {
                        "summary": " / ".join(snippets),
                        "rag_chunks": raw_chunks
                    }
                except Exception as e:
                    print(f"독소조항 로드 오류 ({f_norm}): {e}")

    return toxic_cache

def build_all_products():
    toxic_cache = load_all_toxic_clauses()
    print(f"📦 매핑용 독소조항 문서 로드: {len(toxic_cache)}개")

    # 2_가입제안서는 제외하고 1_가입화면만 탐색
    search_dirs = [
        os.path.join(BASE_DIR, "1_가입화면"),
    ]

    all_products = []
    processed_count = 0

    for base_search in search_dirs:
        if not os.path.exists(base_search):
            print(f"⚠️ 탐색 폴더를 찾을 수 없습니다: {base_search}")
            continue

        for root, _, files in os.walk(base_search):
            for file in files:
                file_norm = nfc(file)
                # 원본 파싱 JSON만 추출 (결과 및 독소 파일 제외)
                if (
                    not file_norm.endswith(".json")
                    or file_norm.endswith("_result.json")
                    or file_norm.endswith("_toxic.json")
                ):
                    continue

                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as jf:
                        data = json.load(jf)
                except Exception:
                    continue

                contract = data.get("contract_summary", {})
                raw_coverages = data.get("coverages", [])
                if not raw_coverages and "results" in data:
                    continue

                p_name = nfc(contract.get("product_name") or file_norm.replace(".json", ""))
                p_years = parse_int_years(contract.get("payment_period", 20), 20)
                m_age = parse_int_years(contract.get("maturity", 90), 90)
                e_age = parse_age(contract.get("age"), contract.get("age_raw"))
                gender = nfc(contract.get("gender", "상관없음"))
                job = nfc(contract.get("job", "상관없음"))
                tier = nfc(contract.get("tier", "표준"))

                # 독소조항 매칭 로직
                matched_summary = ""
                matched_rag_chunks = []
                file_path_norm = nfc(file_path)
                for tkey, tinfo in toxic_cache.items():
                    tokens = [t for t in tkey.split() if len(t) >= 2]
                    if any(tok in file_path_norm or tok in p_name for tok in tokens):
                        matched_summary = tinfo["summary"]
                        matched_rag_chunks = tinfo["rag_chunks"]
                        break
                if not matched_summary and toxic_cache:
                    first_val = list(toxic_cache.values())[0]
                    matched_summary = first_val["summary"]
                    matched_rag_chunks = first_val["rag_chunks"]

                riders = []
                for c in raw_coverages:
                    if c.get("is_enrolled") is False:
                        continue

                    c_name = nfc(c.get("coverage_name") or c.get("coverage_name_raw", ""))
                    if not c_name:
                        continue

                    amt = extract_amount(c.get("subscribed_amount"))
                    prem = c.get("premium", 0) or 0
                    try:
                        prem = int(prem)
                    except:
                        prem = 0

                    riders.append({
                        "coverage_name": c_name,
                        "subscribed_amount": amt,
                        "premium": prem,
                        "payment_period": f"{p_years}년납",
                        "payment_condition": nfc(c.get("payment_condition", "")),
                        "toxic_clauses": matched_summary,
                    })

                if not riders:
                    continue

                display_title = f"{p_name} ({job} | {tier} | {e_age}세 {gender})"
                prod_hash_id = hashlib.md5(f"{display_title}_{file_norm}".encode("utf-8")).hexdigest()[:12]

                all_products.append({
                    "product_id": f"PROD_{prod_hash_id}",
                    "product_name": display_title,
                    "company": nfc(contract.get("company", "DB손해보험")),
                    "job_category": job,
                    "gender": gender,
                    "tier": tier,
                    "payment_years": p_years,
                    "entry_age": e_age,
                    "maturity_age": m_age,
                    "conf_n": 0.95,
                    "riders": riders,
                    "rag_context": matched_rag_chunks[:15]
                })
                processed_count += 1

    out_file = os.path.join(BASE_DIR, "products.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_products, f, ensure_ascii=False, indent=2)

    print(f"🎉 [성공] 총 {processed_count}개의 상품이 '{out_file}'에 통합되었습니다!")

if __name__ == "__main__":
    build_all_products()