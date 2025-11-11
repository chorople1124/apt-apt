# train_xgboost_rent.py
# -*- coding: utf-8 -*-
"""
입력 CSV 예시 컬럼: [시군구, 평수, 월세금(만원), 건축년도]
- 특징: 평수(1개)
- 타깃: 월세금(만원)
- 전세(월세=0) 제거 + IQR로 이상치 약하게 트리밍
- XGBoost로 학습/평가
- 모델(pkl) 저장 + 간단 예측 헬퍼 생성
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# xgboost 설치 필요: pip install xgboost
try:
    from xgboost import XGBRegressor
except Exception as e:
    raise SystemExit(
        "xgboost가 설치되지 않았습니다. 설치: pip install xgboost\n"
        f"원인: {e}"
    )

def read_csv_safely(path: str) -> pd.DataFrame:
    # utf-8-sig -> cp949 순서로 시도
    for enc in ["utf-8-sig", "cp949"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    # 그래도 실패하면 기본 인코딩 시도
    return pd.read_csv(path)

def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = read_csv_safely(csv_path)

    # 필요한 컬럼만 체크
    need_cols = ["평수", "월세금(만원)"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {missing}")

    # 숫자 변환
    df["평수"] = pd.to_numeric(df["평수"], errors="coerce")
    df["월세금(만원)"] = pd.to_numeric(df["월세금(만원)"], errors="coerce")

    # 결측 제거 + 전세(0) 제거
    df = df.dropna(subset=["평수", "월세금(만원)"])
    df = df[df["월세금(만원)"] > 0]

    # 이상치(상단) 완만하게 트리밍: IQR
    q1, q3 = df["월세금(만원)"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    df = df[df["월세금(만원)"] <= upper]

    # 극단 평수도 아주 가볍게 컷(선택적) — 데이터 품질 향상 목적
    df = df[(df["평수"] >= 3) & (df["평수"] <= 100)]  # 필요시 조정

    return df

def train_model(X, y) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,            # 환경 호환성 위해 1로 제한
        tree_method="hist",  # 빠른 학습
        objective="reg:squarederror",
    )
    model.fit(X, y)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="서울시_아파트_전월세_요약.csv",
                        help="입력 CSV 경로 (기본: 현재 폴더)")
    parser.add_argument("--out_dir", type=str, default="model_out",
                        help="모델 및 결과 저장 폴더")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="테스트셋 비율 (기본 0.2)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "rent_xgb_model.pkl")
    helper_path = os.path.join(args.out_dir, "xgb_predict_helper.py")

    # 데이터 로드 & 정제
    df = load_and_clean(args.csv)
    X = df[["평수"]].to_numpy(dtype=float)
    y = df["월세금(만원)"].to_numpy(dtype=float)

    # 학습/평가 분할
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    # 학습
    model = train_model(X_tr, y_tr)

    # 평가
    pred = model.predict(X_te)
    mae = mean_absolute_error(y_te, pred)
    rmse = mean_squared_error(y_te, pred, squared=False)
    r2 = r2_score(y_te, pred)

    print("=== 데이터 크기 ===")
    print(f"전체: {len(df):,} | 학습: {len(X_tr):,} | 테스트: {len(X_te):,}\n")
    print("=== 성능(테스트) ===")
    print(f"MAE :  {mae:.2f} 만원")
    print(f"RMSE: {rmse:.2f} 만원")
    print(f"R^2 : {r2:.3f}\n")

    # 예측 예시
    sample = np.array([[10],[15],[20],[25],[30],[35]], dtype=float)
    sample_pred = model.predict(sample)
    print("=== 예측 예시 (평수 -> 월세 만원) ===")
    for p, v in zip(sample.flatten(), sample_pred):
        print(f"{int(p):>2}평 -> {v:.1f} 만원")

    # 저장
    joblib.dump(model, model_path)
    with open(helper_path, "w", encoding="utf-8") as f:
        f.write(f"""# -*- coding: utf-8 -*-
import joblib, numpy as np

MODEL_PATH = r"{model_path}"

def predict_rent(pyeong: float) -> float:
    model = joblib.load(MODEL_PATH)
    X = np.array([[float(pyeong)]], dtype=float)
    return float(model.predict(X)[0])

if __name__ == "__main__":
    for p in [10, 15, 20, 25, 30, 35]:
        print(f"{{p}}평 -> {{predict_rent(p):.1f}} 만원")
""")
    print(f"\n저장 완료 ✅\n- 모델: {model_path}\n- 헬퍼: {helper_path}")

if __name__ == "__main__":
    main()
