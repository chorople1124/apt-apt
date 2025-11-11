# app.py
# -*- coding: utf-8 -*-
import os
import io
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ====== 설정 ======
CSV_PATH = "서울시_아파트_전월세_요약.csv"  # [시군구, 평수, 월세금(만원), 건축년도]
MODEL_PATH = "rent_xgb_model.pkl"
TITLE = "🏢 평수 입력 → 월세 예측"
DESC = "평수(1개 특징)만으로 XGBoost 회귀 모델이 월세(만원)를 예측합니다."

# ====== XGBoost 준비 ======
try:
    from xgboost import XGBRegressor
except Exception as e:
    st.stop()  # UI에서 에러 안내
    raise

st.set_page_config(page_title="평수→월세 예측", layout="centered")
st.title(TITLE)
st.caption(DESC)

# ====== 데이터 로딩 & 정제 ======
@st.cache_data(show_spinner=False)
def read_csv_safely(path: str) -> pd.DataFrame:
    tried = []
    for enc in ["utf-8-sig", "cp949"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            tried.append(f"{enc}: {e}")
    # 마지막 시도
    return pd.read_csv(path)

@st.cache_data(show_spinner=True)
def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = read_csv_safely(csv_path)
    for c in ["평수", "월세금(만원)"]:
        if c not in df.columns:
            raise ValueError(f"CSV에 '{c}' 컬럼이 필요합니다.")
    df["평수"] = pd.to_numeric(df["평수"], errors="coerce")
    df["월세금(만원)"] = pd.to_numeric(df["월세금(만원)"], errors="coerce")
    df = df.dropna(subset=["평수", "월세금(만원)"])
    df = df[df["월세금(만원)"] > 0]  # 전세 제거
    # 가벼운 이상치 컷(IQR 상단)
    q1, q3 = df["월세금(만원)"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    df = df[df["월세금(만원)"] <= upper]
    # 극단 평수 컷(옵션)
    df = df[(df["평수"] >= 3) & (df["평수"] <= 100)]
    return df

# ====== 모델 로드/학습 ======
@st.cache_resource(show_spinner=True)
def get_or_train_model(csv_path: str, model_path: str):
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            return model
        except Exception:
            pass  # 모델 불러오기 실패 시 재학습
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

# 파일 업로드(선택) 또는 기본 CSV 사용
st.write("### 데이터 선택")
up = st.file_uploader("CSV 업로드(선택) — 업로드 시 해당 파일로 학습합니다.", type=["csv"])
if up is not None:
    # 업로드된 파일로 임시 모델 학습 (디스크 저장 안 함)
    df_tmp = pd.read_csv(up)
    # 업로드 파일도 정제 함수 재사용
    def _save_and_train_from_uploaded(df_csv: pd.DataFrame):
        # 임시 경로에 저장 후 동일 파이프라인 사용
        tmp_path = "_uploaded.csv"
        df_csv.to_csv(tmp_path, index=False, encoding="utf-8-sig")
        # 캐시 무효화를 위해 seed 파라미터를 넣어 호출
        model = get_or_train_model(tmp_path, MODEL_PATH + ".tmp")
        return model, tmp_path
    try:
        # 업로드 파일을 정제 함수가 기대하는 형식으로 저장 후 재사용
        tmp_path = "_uploaded.csv"
        df_tmp.to_csv(tmp_path, index=False, encoding="utf-8-sig")
        model = get_or_train_model(tmp_path, MODEL_PATH + ".tmp")
        data_path_in_use = "업로드 파일"
    except Exception as e:
        st.error(f"업로드 파일 처리 중 오류: {e}")
        st.stop()
else:
    # 기본 파일 사용
    if not os.path.exists(CSV_PATH):
        st.error(f"기본 CSV가 없습니다: {CSV_PATH}")
        st.stop()
    model = get_or_train_model(CSV_PATH, MODEL_PATH)
    data_path_in_use = CSV_PATH

st.success(f"모델 준비 완료 ✅  (데이터: {data_path_in_use})")

# ====== 예측 폼 ======
st.markdown("### 🔮 월세 예측 폼")
with st.form("predict_form"):
    # 평수 입력 슬라이더/숫자
    default_p = 25.0
    pyeong = st.number_input("평수", min_value=3.0, max_value=100.0, value=default_p, step=0.5)
    submitted = st.form_submit_button("예측하기")

if submitted:
    X_row = np.array([[float(pyeong)]], dtype=float)
    y_hat = float(model.predict(X_row)[0])
    st.subheader("예측 결과")
    st.metric(label=f"{pyeong:.1f}평 예상 월세", value=f"{y_hat:.1f} 만원")

st.divider()
st.caption("주의: 평수 1개 특징만 사용하므로 지역·연식 등은 반영되지 않습니다. "
           "필요 시 다변수(건축년도, 시군구 등) 버전으로 확장해 드릴게요.")
