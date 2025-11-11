# app.py
# -*- coding: utf-8 -*-
import io
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# XGBoost 로더
USE_XGB = True
XGB_IMPORT_ERROR = ""
try:
    from xgboost import XGBRegressor
except Exception as e:
    USE_XGB = False
    XGB_IMPORT_ERROR = str(e)

DEFAULT_PATH = "서울시_아파트_전월세_요약.csv"  # [시군구, 평수, 월세금(만원), 건축년도]

st.set_page_config(page_title="평수→월세 예측 (XGBoost)", layout="wide")
st.title("🏢 서울 아파트 평수 → 월세 예측 (XGBoost)")

with st.sidebar:
    st.header("데이터 불러오기")
    uploaded = st.file_uploader("CSV 업로드 (예: 서울시_아파트_전월세_요약.csv)", type=["csv"])
    use_default = st.checkbox("기본 경로 사용", value=not uploaded, help=f"현재 폴더의 '{DEFAULT_PATH}' 사용")

    st.markdown("---")
    st.subheader("전처리 옵션")
    drop_zero = st.checkbox("전세(월세=0) 제거", value=True)
    trim_outliers = st.checkbox("이상치 트리밍(IQR 상단 1.5)", value=True)
    min_pyeong, max_pyeong = st.slider("평수 범위 필터", 3.0, 120.0, (3.0, 100.0), 0.5)
    test_size = st.slider("테스트셋 비율", 0.05, 0.4, 0.2, 0.05)

    st.markdown("---")
    st.subheader("특징(Feature) 선택")
    use_only_pyeong = st.checkbox("평수만 사용 (기본)", value=True)
    use_year = st.checkbox("건축년도 포함", value=False, disabled=use_only_pyeong)
    use_region = st.checkbox("시군구 포함(원-핫 인코딩)", value=False, disabled=use_only_pyeong)

    st.markdown("---")
    st.subheader("모델 설정")
    n_estimators = st.slider("n_estimators", 50, 600, 300, 50)
    max_depth = st.slider("max_depth", 2, 12, 4, 1)
    learning_rate = st.select_slider("learning_rate", options=[0.03, 0.05, 0.08, 0.1, 0.2], value=0.08)
    reg_lambda = st.select_slider("reg_lambda", options=[0.0, 0.5, 1.0, 2.0, 5.0], value=1.0)
    train_btn = st.button("🔁 모델 학습 / 재학습")

def read_csv_safely(file_or_path):
    tried = []
    for enc in ["utf-8-sig", "cp949"]:
        try:
            return pd.read_csv(file_or_path, encoding=enc)
        except Exception as e:
            tried.append(f"{enc}: {e}")
            continue
    return pd.read_csv(file_or_path)

if uploaded is not None:
    df_raw = read_csv_safely(uploaded)
elif use_default and os.path.exists(DEFAULT_PATH):
    df_raw = read_csv_safely(DEFAULT_PATH)
else:
    st.warning("CSV를 업로드하거나 '기본 경로 사용'을 체크해 주세요.")
    st.stop()

expected = ["시군구", "평수", "월세금(만원)", "건축년도"]
missing = [c for c in expected if c not in df_raw.columns]
if missing:
    st.error(f"필수 컬럼이 누락되었습니다: {missing}\nCSV에 다음 컬럼이 있어야 합니다: {expected}")
    st.stop()

df = df_raw.copy()
df["평수"] = pd.to_numeric(df["평수"], errors="coerce")
df["월세금(만원)"] = pd.to_numeric(df["월세금(만원)"], errors="coerce")
df["건축년도"] = pd.to_numeric(df["건축년도"], errors="coerce")

df = df[(df["평수"] >= min_pyeong) & (df["평수"] <= max_pyeong)]
df = df.dropna(subset=["평수", "월세금(만원)"])
if drop_zero:
    df = df[df["월세금(만원)"] > 0]
if trim_outliers and len(df) > 0:
    q1, q3 = df["월세금(만원)"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    df = df[df["월세금(만원)"] <= upper]

st.success(f"데이터 준비 완료: {len(df):,}건")
with st.expander("데이터 미리보기"):
    st.dataframe(df.head(20))

feature_cols = ["평수"]
if not use_only_pyeong:
    if use_year:
        feature_cols.append("건축년도")
    if use_region:
        feature_cols.append("시군구")

X_df = df[feature_cols].copy()
y = df["월세금(만원)"].values

numeric_features = [c for c in feature_cols if c != "시군구"]
categorical_features = ["시군구"] if "시군구" in feature_cols else []

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ],
    remainder="drop",
)

if not USE_XGB:
    st.error("xgboost가 설치되어 있지 않습니다. 터미널에서 `pip install xgboost` 후 다시 실행하세요.")
    st.stop()

from xgboost import XGBRegressor
reg = XGBRegressor(
    n_estimators=int(n_estimators),
    max_depth=int(max_depth),
    learning_rate=float(learning_rate),
    subsample=0.9,
    colsample_bytree=1.0,
    reg_lambda=float(reg_lambda),
    random_state=42,
    n_jobs=1,
    tree_method="hist",
    objective="reg:squarederror",
)

model = Pipeline(steps=[("prep", preprocess), ("reg", reg)])

if train_btn or "fitted_" not in st.session_state:
    if len(df) < 10:
        st.warning("데이터가 너무 적어 학습이 어려워요. 최소 10건 이상 권장합니다.")
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=float(test_size), random_state=42
    )
    model.fit(X_train, y_train)
    st.session_state["fitted_"] = True
    st.session_state["model"] = model
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test
    st.toast("모델 학습 완료!", icon="✅")

if "model" not in st.session_state:
    st.stop()

model = st.session_state["model"]
X_test = st.session_state["X_test"]
y_test = st.session_state["y_test"]
pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred, squared=False)
r2 = r2_score(y_test, pred)

col1, col2, col3 = st.columns(3)
col1.metric("MAE (만원)", f"{mae:.2f}")
col2.metric("RMSE (만원)", f"{rmse:.2f}")
col3.metric("R²", f"{r2:.3f}")

st.subheader("실제 vs 예측 (테스트셋)")
fig = plt.figure()
plt.scatter(y_test, pred, alpha=0.6)
plt.xlabel("실제 월세(만원)")
plt.ylabel("예측 월세(만원)")
plt.title("실제 vs 예측")
st.pyplot(fig)

st.markdown("---")
st.header("🔮 월세 예측")

inp_p = st.slider("평수", float(df["평수"].min()), float(df["평수"].max()), float(np.median(df["평수"])), 0.5)

extra = {}
if "건축년도" in feature_cols:
    yr_min = int(np.nan_to_num(df["건축년도"].min(), nan=1990))
    yr_max = int(np.nan_to_num(df["건축년도"].max(), nan=2025))
    extra["건축년도"] = st.number_input("건축년도", min_value=1900, max_value=2100, value=min(max(yr_min, 1990), yr_max))

if "시군구" in feature_cols:
    regions = sorted(df["시군구"].dropna().unique().tolist())
    extra["시군구"] = st.selectbox("시군구", options=regions, index=0 if regions else None)

def build_input_row(pyeong: float, extras: dict) -> pd.DataFrame:
    row = {"평수": float(pyeong)}
    for k in ["건축년도", "시군구"]:
        if k in feature_cols:
            if k == "건축년도":
                row[k] = extras.get(k, int(np.nan_to_num(df["건축년도"].median(), nan=2005)))
            if k == "시군구":
                row[k] = extras.get(k, df["시군구"].mode().iloc[0] if not df["시군구"].empty else "")
    return pd.DataFrame([row], columns=feature_cols)

if st.button("예측 실행"):
    X_row = build_input_row(inp_p, extra)
    y_hat = float(model.predict(X_row)[0])
    st.success(f"예측 월세: **{y_hat:.1f} 만원**")

st.markdown("---")
st.subheader("모델 내보내기")
bytes_buf = io.BytesIO()
joblib.dump(model, bytes_buf)
st.download_button("💾 학습 모델 다운로드 (.pkl)", data=bytes_buf.getvalue(), file_name="rent_xgb_model.pkl")

st.caption("Tip: 평수 하나만으로는 지역·연식 효과를 반영하기 어려워 R²가 낮을 수 있어요. "
           "사이드바에서 '건축년도', '시군구'를 추가하면 성능이 개선됩니다.")
