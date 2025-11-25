# app.py
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import xgboost as xgb   # ← sklearn 필요 없는 native XGBoost 사용

# ==========================
# 🔥 한글 폰트 깨짐 방지 설정
# ==========================
def set_korean_font():
    font_candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/Library/Fonts/AppleSDGothicNeo.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf",
    ]

    chosen = None
    for font_path in font_candidates:
        if os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)
            family_name = os.path.basename(font_path).split(".")[0]
            mpl.rc("font", family=family_name)
            chosen = font_path
            break

    if chosen is None:
        mpl.rc("font", family="sans-serif")

    mpl.rcParams["axes.unicode_minus"] = False

set_korean_font()

CSV_PATH = "csv.csv"
MODEL_PATH = "rent_xgb_model.json"   # ← Booster 저장 방식은 pkl이 아니라 json 추천

st.set_page_config(page_title="서울 아파트 월세 예측", layout="centered")
st.title("🏙️ 서울 아파트 월세 예측기")
st.caption("전용 CSV를 이용해 평수(1개 변수)로 월세(만원)을 예측합니다.")

# ==========================
# CSV 불러오기
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
    df = df[df["월세금(만원)"] > 0]

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

    # Booster 대신 DMatrix 사용
    dtrain = xgb.DMatrix(X, label=y)

    # 기존 모델 있으면 로드
    if os.path.exists(MODEL_PATH):
        try:
            booster = xgb.Booster()
            booster.load_model(MODEL_PATH)
            return booster
        except:
            pass

    params = {
        "eta": 0.08,
        "max_depth": 4,
        "subsample": 0.9,
        "colsample_bytree": 1.0,
        "lambda": 1.0,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "seed": 42
    }

    num_round = 300
    booster = xgb.train(params, dtrain, num_boost_round=num_round)

    booster.save_model(MODEL_PATH)
    return booster

df = load_data()
model = train_or_load_model()
st.success("모델 준비 완료 ✅")

st.subheader("📄 데이터 미리보기")
st.dataframe(df.head())

st.divider()

# ==========================
# 📏 평수 입력 → 예측
# ==========================
st.subheader("📏 평수 입력")

default_pyeong = float(df["평수"].median())
pyeong = st.number_input(
    "평수 입력",
    min_value=3.0,
    max_value=100.0,
    value=default_pyeong,
    step=0.5
)

dpred = xgb.DMatrix(np.array([[pyeong]]))
pred_for_input = float(model.predict(dpred)[0])

if st.button("예상 월세 예측하기"):
    st.metric(label=f"{pyeong:.1f}평 예상 월세", value=f"{pred_for_input:.1f} 만원")

st.divider()

# ==========================
# 📊 입력한 평수를 기준으로 한 꺾은선 그래프
# ==========================
st.subheader("📊 입력 평수를 기준으로 한 예측 월세 꺾은선 그래프")

if len(df) > 0:
    min_p = float(df["평수"].min())
    max_p = float(df["평수"].max())

    p_start = max(min_p, pyeong - 10)
    p_end = min(max_p, pyeong + 10)

    if p_start >= p_end:
        p_start, p_end = min_p, max_p

    p_range = np.linspace(p_start, p_end, 100).reshape(-1, 1)
    pred_range = model.predict(xgb.DMatrix(p_range))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p_range, pred_range, linewidth=2)

    ax.scatter([pyeong], [pred_for_input], s=60)
    ax.axvline(pyeong, linestyle="--")

    ax.set_title(f"{pyeong:.1f}평을 기준으로 한 예측 월세 곡선")
    ax.set_xlabel("평수")
    ax.set_ylabel("월세 (만원)")
    ax.grid(True)

    st.pyplot(fig)

st.divider()
st.caption(
    "⚙️ 참고: 현재 모델은 평수만을 고려하여 예측합니다. "
    "건축년도·지역 등의 변수를 추가하면 정확도를 더 높일 수 있습니다."
)
