# app.py
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ===== 설정 =====
CSV_PATH = "서울시_아파트_전월세_요약.csv"   # 컬럼: [시군구, 평수, 월세금(만원), 건축년도]
MODEL_PATH = "rent_xgb_model.pkl"

st.set_page_config(page_title="평수 → 월세 예측", layout="centered")
st.title("🏢 평수만 입력 → 월세 예측 (XGBoost)")
st.caption("평수 1개 특징으로 월세(만원)을 예측합니다.")

# ===== XGBoost =====
from xgboost import XGBRegressor

# ===== 유틸 =====
def read_csv_safely(path: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "cp949", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    # 마지막 시도(인코딩 지정 X)
    return pd.read_csv(path)

@st.cache_data(show_spinner=True)
def load_and_clean(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV가 없습니다: {csv_path}")
    df = read_csv_safely(csv_path)

    # 필요한 컬럼 체크
    for c in ["평수", "월세금(만원)"]:
        if c not in df.columns:
            raise ValueError(f"CSV에 '{c}' 컬럼이 필요합니다.")

    # 숫자 변환 + 전처리
    df["평수"] = pd.to_numeric(df["평수"], errors="coerce")
    df["월세금(만원)"] = pd.to_numeric(df["월세금(만원)"], errors="coerce")
    df = df.dropna(subset=["평수", "월세금(만원)"])
    df = df[df["월세금(만원)"] > 0]             # 전세 제거
    # 가벼운 이상치 제거(IQR 상단)
    q1, q3 = df["월세금(만원)"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    df = df[df["월세금(만원)"] <= upper]
    # 극단 평수 컷(옵션)
    df = df[(df["평수"] >= 3) & (df["평수"] <= 100)]
    return df

@st.cache_resource(show_spinner=True)
def train_or_load_model(csv_path: str, model_path: str):
    # 모델 파일 있으면 로드, 없으면 학습 후 저장
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception:
            pass
    df = load_and_clean(csv_path)
    X = df[["평수"]].to_numpy(dtype=float)
    y = df["월세금(만원)"].to_numpy(dtype=float)

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
        tree_method="hist",
        objective="reg:squarederror",
    )
    model.fit(X, y)
    joblib.dump(model, model_path)
    return model

# ===== 모델 준비 =====
try:
    model = train_or_load_model(CSV_PATH, MODEL_PATH)
    st.success("모델 준비 완료 ✅")
except Exception as e:
    st.error(f"모델 준비 중 오류: {e}")
    st.stop()

# ===== 예측 폼 =====
st.markdown("### 🔮 월세 예측")
default_p = 25.0
pyeong = st.number_input("평수", min_value=3.0, max_value=100.0, value=default_p, step=0.5)
if st.button("예측하기"):
    X_row = np.array([[float(pyeong)]], dtype=float)
    y_hat = float(model.predict(X_row)[0])
    st.metric(label=f"{pyeong:.1f}평 예상 월세", value=f"{y_hat:.1f} 만원")

st.caption("참고: 평수만 사용했기 때문에 지역·연식 영향은 반영되지 않습니다. 필요하면 다변수(건축년도·시군구)로 확장해 드릴게요.")
