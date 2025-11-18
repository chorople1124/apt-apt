# app.py
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# ==========================
# 기본 설정
# ==========================
CSV_PATH = "csv.csv"   # CSV와 app.py가 같은 폴더에 있다고 가정
MODEL_PATH = "rent_xgb_model.pkl"

st.set_page_config(page_title="서울 아파트 월세 예측", layout="centered")
st.title("🏙️ 서울 아파트 월세 예측기")
st.caption("전용 CSV를 이용해 평수(1개 변수)로 월세(만원)을 예측합니다.")

# ==========================
# CSV 불러오기 및 전처리
# ==========================
@st.cache_data
def load_data():
    if not os.path.exists(CSV_PATH):
        st.error(f"CSV 파일이 존재하지 않습니다: {CSV_PATH}")
        st.stop()
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    if "평수" not in df.columns or "월세금(만원)" not in df.columns:
        st.error("CSV 파일에 '평수'와 '월세금(만원)' 컬럼이 있어야 합니다.")
        st.stop()

    df["평수"] = pd.to_numeric(df["평수"], errors="coerce")
    df["월세금(만원)"] = pd.to_numeric(df["월세금(만원)"], errors="coerce")
    df = df.dropna(subset=["평수", "월세금(만원)"])
    df = df[df["월세금(만원)"] > 0]  # 전세 제거

    # 이상치 간단 제거 (IQR)
    q1, q3 = df["월세금(만원)"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    df = df[df["월세금(만원)"] <= upper]
    return df

# ==========================
# 모델 학습 or 불러오기
# ==========================
@st.cache_resource
def train_or_load_model():
    df = load_data()
    X = df[["평수"]].to_numpy(dtype=float)
    y = df["월세금(만원)"].to_numpy(dtype=float)

    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            return model
        except Exception:
            pass

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
    joblib.dump(model, MODEL_PATH)
    return model

df = load_data()
model = train_or_load_model()
st.success("모델 준비 완료 ✅")

# ==========================
# 📊 평수 vs 월세 그래프
# ==========================
st.subheader("📊 평수에 따른 월세 분포 및 예측 곡선")

if len(df) > 0:
    fig, ax = plt.subplots(figsize=(8, 5))

    # 실제 데이터 산점도
    ax.scatter(df["평수"], df["월세금(만원)"], alpha=0.5, label="실제 데이터")

    # 모델 예측 곡선 (평수 범위 전체에 대해)
    min_p = df["평수"].min()
    max_p = df["평수"].max()
    p_range = np.linspace(min_p, max_p, 100).reshape(-1, 1)
    pred_range = model.predict(p_range)

    ax.plot(p_range, pred_range, linewidth=2, label="모델 예측 곡선")

    ax.set_title("평수에 따른 월세(만원)")
    ax.set_xlabel("평수")
    ax.set_ylabel("월세 (만원)")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)
else:
    st.info("그래프를 그릴 데이터가 충분하지 않습니다.")

st.divider()

# ==========================
# 평수 입력 → 예측
# ==========================
st.subheader("📏 평수 입력")

# 기본값: 데이터의 중앙값 사용 (없으면 25평)
default_pyeong = 25.0
if "평수" in df.columns and df["평수"].notna().sum() > 0:
    default_pyeong = float(df["평수"].median())

pyeong = st.number_input(
    "평수 입력",
    min_value=3.0,
    max_value=100.0,
    value=default_pyeong,
    step=0.5
)

if st.button("예상 월세 예측하기"):
    X_input = np.array([[pyeong]])
    pred = float(model.predict(X_input)[0])
    st.metric(label=f"{pyeong:.1f}평 예상 월세", value=f"{pred:.1f} 만원")

st.divider()
st.caption(
    "⚙️ 참고: 현재 모델은 평수만을 고려하여 예측합니다. "
    "건축년도·지역 등의 변수를 추가하면 정확도를 더 높일 수 있습니다."
)
