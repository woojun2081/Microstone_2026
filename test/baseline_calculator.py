import pandas as pd
import numpy as np

# 1. 데이터 로드 및 병합
df_stat = pd.read_csv("데이터 2차 정제 결과.csv")
df_cat = pd.read_csv("데이터 2차 정제 결과_대분류.csv")
df_merged = pd.merge(df_stat, df_cat, on=["국민관심진료행위코드", "국민관심진료행위명"], how="left")

# 총 발생 의료비(Expected Risk Burden) 산출
df_merged["총진료부담액"] = df_merged["환자수"] * df_merged["1인당_평균진료금"]


def get_age_group(age: int) -> str:
    """나이를 CSV의 5세 연령대 문자열로 변환"""
    if age < 5: return "1_4세"
    elif age >= 80: return "80세 이상"
    else:
        lower = (age // 5) * 5
        upper = lower + 4
        return f"{lower}_{upper}세"


def get_user_baseline(gender: str, age: int):
    """
    사용자의 성별(남/여/미지정)과 나이(정수)를 받아
    UI 슬라이더의 초기 디폴트값(0.0 ~ 1.0)을 계산
    """
    age_str = get_age_group(age)
    
    # 성별 필터링 (미지정 시 전체 평균)
    if gender in ["남", "여"]:
        subset = df_merged[(df_merged["성별구분"] == gender) & (df_merged["5세연령대"] == age_str)]
    else:
        subset = df_merged[df_merged["5세연령대"] == age_str]

    if subset.empty:
        return {"cancer": 0.25, "vascular": 0.25, "injury": 0.25, "treatment": 0.25}

    # 대분류별 총 진료비 부담액 집계
    cat_totals = subset.groupby("대분류")["총진료부담액"].sum()

    cancer_cost = cat_totals.get("암", 0)
    vascular_cost = cat_totals.get("순환계 질환", 0) + cat_totals.get("신경계 질환", 0)
    injury_cost = cat_totals.get("상해", 0)
    general_disease = cat_totals.get("질병", 0)

    total_cost = cancer_cost + vascular_cost + injury_cost + general_disease
    if total_cost == 0: total_cost = 1

    # 의료 서비스 유형별(입원/수술) 비중
    service_totals = subset.groupby("의료 서비스")["총진료부담액"].sum()
    inpatient_ratio = (service_totals.get("입원", 0) + service_totals.get("수술", 0)) / max(service_totals.sum(), 1)

    # 0.0 ~ 1.0 상대 비중 산출
    return {
        "age_group": age_str,
        "gender": gender,
        "cancer_baseline": round(cancer_cost / total_cost, 4),        # 암 슬라이더 디폴트
        "vascular_baseline": round(vascular_cost / total_cost, 4),    # 뇌·심장 슬라이더 디폴트
        "injury_baseline": round(injury_cost / total_cost, 4),        # 상해 슬라이더 디폴트
        "inpatient_baseline": round(inpatient_ratio, 4)               # 입원/수술 슬라이더 디폴트
    }


# 테스트 실행: 22세 남성과 52세 여성 비교
print("🧑 22세 남성 디폴트:", get_user_baseline("남", 22))
print("👩 52세 여성 디폴트:", get_user_baseline("여", 52))