# app.py
# Advanced Stock Price Forecaster (Realtime Yahoo Finance + Normality & ADF + Diff)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from scipy.stats import shapiro
from statsmodels.tsa.stattools import adfuller

# ============================
# إعداد الصفحة
# ============================
st.set_page_config(
    page_title="Advanced Stock Price Forecaster",
    layout="wide"
)

# ============================
# دوال المساعدة
# ============================

def load_stock_data(ticker: str, years: int):
    """
    يجلب السعر الحالي + البيانات التاريخية من Yahoo Finance.
    """
    stock = yf.Ticker(ticker)

    # سعر لحظي
    info = getattr(stock, "fast_info", {}) or {}
    current_price = info.get("last_price", None)

    # احتياط لو fast_info ما أعطى سعر
    if current_price is None:
        hist = stock.history(period="1d")
        if not hist.empty:
            current_price = float(hist["Close"].iloc[-1])
        else:
            current_price = np.nan

    # بيانات تاريخية لعدد من السنوات
    df = stock.history(period=f"{years}y")
    df = df.dropna()

    return current_price, df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    مؤشرات فنية أساسية: SMA, EMA, RSI, MACD.
    """
    df = df.copy()

    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    إضافة Lags + Returns + Volatility.
    """
    df = df.copy()

    # Lags للأسعار
    for lag in [1, 2, 3, 5, 7, 10, 14]:
        df[f"Close_lag_{lag}"] = df["Close"].shift(lag)

    # Returns
    df["Return_1"] = df["Close"].pct_change(1)
    df["Return_3"] = df["Close"].pct_change(3)
    df["Return_7"] = df["Close"].pct_change(7)

    # Volatility
    df["Volatility_7"] = df["Return_1"].rolling(7).std()
    df["Volatility_14"] = df["Return_1"].rolling(14).std()

    return df


def check_normality(series: pd.Series):
    """
    اختبار Shapiro-Wilk للتوزيع الطبيعي.
    H0: البيانات تتبع التوزيع الطبيعي.
    """
    s = series.dropna()
    # Shapiro لا يحب العينات الضخمة جداً
    if len(s) > 5000:
        s = s.sample(5000, random_state=42)

    stat, p = shapiro(s)
    return {
        "statistic": stat,
        "p_value": p,
        "is_normal": p > 0.05
    }


def check_stationarity(series: pd.Series):
    """
    اختبار ADF للثبات.
    H0: السلسلة غير ثابتة (يوجد جذر واحد).
    """
    s = series.dropna()
    try:
        result = adfuller(s)
        adf_stat, p, used_lag, nobs, crit_vals, icbest = result
        is_stationary = p < 0.05
    except Exception:
        # في حال فشل الـ ADF لأي سبب
        adf_stat, p, is_stationary = np.nan, np.nan, False

    return {
        "ADF Statistic": adf_stat,
        "p_value": p,
        "is_stationary": is_stationary
    }


def apply_log_transform(series: pd.Series):
    """
    تحويل Log (مع مراعاة القيم الموجبة).
    """
    return np.log(series)


def apply_differencing(series: pd.Series, order: int = 1):
    """
    Differencing من الدرجة الأولى افتراضياً.
    """
    return series.diff(order)


def build_dataset(df: pd.DataFrame, feature_cols, horizon: int):
    """
    تجهيز داتا لتدريب نموذج يتنبأ Close بعد عدد أيام (horizon).
    """
    df2 = df.copy()
    target_col = f"target_{horizon}"

    # الهدف: سعر الإغلاق بعد horizon يوم
    df2[target_col] = df2["Close"].shift(-horizon)

    df2 = df2.dropna(subset=feature_cols + [target_col])

    if len(df2) < 100:
        return None

    X = df2[feature_cols]
    y = df2[target_col]

    n = len(df2)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    return {
        "X_train": X.iloc[:train_end],
        "y_train": y.iloc[:train_end],
        "X_val": X.iloc[train_end:val_end],
        "y_val": y.iloc[train_end:val_end],
        "X_test": X.iloc[val_end:],
        "y_test": y.iloc[val_end:],
    }


def train_models_for_horizon(dataset, feature_cols):
    """
    تدريب 3 نماذج لكل أفق زمني.
    """
    models_def = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    metrics_list = []
    trained = {}

    X_train = dataset["X_train"]
    y_train = dataset["y_train"]
    X_test = dataset["X_test"]
    y_test = dataset["y_test"]

    for name, base_model in models_def.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", base_model),
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # اتجاه السعر (طلوع / نزول)
        close_test = X_test["Close"].values
        dir_real = np.sign(y_test.values - close_test)
        dir_pred = np.sign(y_pred - close_test)
        directional_acc = float((dir_real == dir_pred).mean() * 100)

        metrics_list.append({
            "Model": name,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "Directional_Accuracy": directional_acc,
        })

        trained[name] = {"pipeline": pipe, "rmse": rmse}

    metrics_df = pd.DataFrame(metrics_list).sort_values(
        "Directional_Accuracy", ascending=False
    )
    return trained, metrics_df

# ============================
# الإعدادات – Sidebar
# ============================

with st.sidebar:
    st.title("⚙️ Configuration")

    market = st.selectbox(
        "Select Market",
        ["Saudi Stocks (Tadawul - TASI)", "US Stocks", "Crypto"],
        index=0,
    )

    default_ticker = "2010.SR" if market == "Saudi Stocks (Tadawul - TASI)" else "AAPL"
    ticker = st.text_input("Enter Stock Ticker", default_ticker)

    years_hist = st.slider("Years of Historical Data", 1, 10, 5)

    horizons_labels = ["7 Days", "14 Days", "30 Days"]
    horizons_selected_labels = st.multiselect(
        "Forecast Horizons", horizons_labels, default=horizons_labels
    )
    horizon_map = {"7 Days": 7, "14 Days": 14, "30 Days": 30}
    horizons = [horizon_map[h] for h in horizons_selected_labels]

    train_button = st.button("🚀 Train Models & Forecast")

    st.markdown("---")
    st.subheader("Notes")
    st.markdown(
        """
        • يتم فحص التوزيع الطبيعي للسلسلة (Shapiro).  
        • إذا غير طبيعية → يتم تطبيق Log.  
        • يتم فحص الثبات (ADF Test).  
        • إذا غير ثابتة → يتم تطبيق Differencing.  
        • كل ذلك قبل بناء ميزات النماذج.
        """
    )

# ============================
# Tabs
# ============================

st.title("📈 Advanced Stock Price Forecaster")

tab_summary, tab_forecasts, tab_models = st.tabs(
    ["Pipeline Summary", "Future Price Forecasts", "Model Performance"]
)

if not train_button:
    with tab_summary:
        st.info("⬅️ اختر الإعدادات من القائمة الجانبية ثم اضغط **Train Models & Forecast**.")
    with tab_forecasts:
        st.warning("لم يتم تدريب النماذج بعد.")
    with tab_models:
        st.warning("لم يتم تدريب النماذج بعد.")
else:
    # ============================
    # 1) جلب البيانات
    # ============================
    current_price, df_raw = load_stock_data(ticker, years_hist)

    if df_raw.empty:
        st.error("لم يتم جلب أي بيانات من Yahoo Finance. تحققي من الرمز أو عدد السنوات.")
    else:
        # ============================
        # 2) مؤشرات فنية + Lags
        # ============================
        df_feat = add_technical_indicators(df_raw)
        df_feat = add_lag_features(df_feat)

        # ============================
        # 3) Normality + Log Transform
        # ============================
        normal_res = check_normality(df_feat["Close"])
        use_log = not normal_res["is_normal"]

        if use_log:
            df_feat["Close_log"] = apply_log_transform(df_feat["Close"])
            series_for_adf = df_feat["Close_log"]
        else:
            df_feat["Close_log"] = apply_log_transform(df_feat["Close"])
            series_for_adf = df_feat["Close"]

        # ============================
        # 4) ADF + Differencing
        # ============================
        adf_res = check_stationarity(series_for_adf)
        use_diff = not adf_res["is_stationary"]

        if use_diff:
            df_feat["Close_diff1"] = apply_differencing(series_for_adf)
        else:
            # حتى لو ثابتة نضيف diff كميزة (تعبّر عن التغير)
            df_feat["Close_diff1"] = apply_differencing(series_for_adf)

        # أسقط الصفوف الناقصة بسبب rolling/diff/log
        df_feat = df_feat.dropna()

        # ============================
        # 5) تعريف الميزات للنماذج
        # ============================
        feature_cols = [
            # السعر والحجم
            "Close", "Volume",
            # مؤشرات فنية
            "SMA_20", "SMA_50",
            "EMA_10", "EMA_20",
            "RSI", "MACD", "Signal",
            # Log + Diff
            "Close_log", "Close_diff1",
            # Lags
            "Close_lag_1", "Close_lag_2", "Close_lag_3",
            "Close_lag_5", "Close_lag_7", "Close_lag_10", "Close_lag_14",
            # Returns
            "Return_1", "Return_3", "Return_7",
            # Volatility
            "Volatility_7", "Volatility_14",
        ]

        # تأكد كل الأعمدة موجودة
        feature_cols = [c for c in feature_cols if c in df_feat.columns]

        results_by_h = {}
        models_by_h = {}
        forecasts = {}

        # ============================
        # 6) تدريب النماذج لكل أفق
        # ============================
        for h in horizons:
            dataset = build_dataset(df_feat, feature_cols, horizon=h)
            if dataset is None:
                continue

            trained, df_metrics = train_models_for_horizon(dataset, feature_cols)
            results_by_h[h] = df_metrics
            models_by_h[h] = trained

            # توقعات لكل نموذج من أفضل 3
            forecasts[h] = {}
            for model_name in df_metrics["Model"].head(3):
                pipe = trained[model_name]["pipeline"]
                rmse = trained[model_name]["rmse"]

                X_last = df_feat[feature_cols].iloc[[-1]]
                pred = float(pipe.predict(X_last)[0])

                diff_pct = (pred - current_price) / current_price * 100 if current_price else np.nan
                low = pred - rmse
                high = pred + rmse

                if diff_pct > 1:
                    sentiment = "Bullish"
                elif diff_pct < -1:
                    sentiment = "Bearish"
                else:
                    sentiment = "Neutral"

                forecasts[h][model_name] = {
                    "forecast": pred,
                    "diff_pct": diff_pct,
                    "low": low,
                    "high": high,
                    "sentiment": sentiment
                }

        if not results_by_h:
            with tab_summary:
                st.error("البيانات غير كافية لتدريب النماذج على الآفاق الزمنية المختارة.")
        else:
            # نختار أفق رئيسي للعرض (يفضل 7)
            primary_h = 7 if 7 in results_by_h else sorted(results_by_h.keys())[0]
            primary_df = results_by_h[primary_h]

            # ============================
            # TAB 1 – Pipeline Summary
            # ============================
            with tab_summary:
                st.subheader("1️⃣ Data Collection")
                st.success(
                    f"Loaded {len(df_raw)} rows for **{ticker}** "
                    f"in market **{market}**. Current Price: **{current_price:.2f}**"
                )

                st.subheader("2️⃣ Normality Check (Shapiro-Wilk)")
                st.write(f"Statistic: `{normal_res['statistic']:.4f}`, p-value: `{normal_res['p_value']:.4f}`")
                if normal_res["is_normal"]:
                    st.success("✔️ السلسلة تقريباً تتبع التوزيع الطبيعي (لم يتم فرض Log في النموذج).")
                else:
                    st.warning("❌ السلسلة لا تتبع التوزيع الطبيعي. تم إنشاء متغير Log (Close_log) واستخدامه كميزة.")

                st.subheader("3️⃣ Stationarity Check (ADF Test)")
                st.write(f"ADF Statistic: `{adf_res['ADF Statistic']:.4f}`, p-value: `{adf_res['p_value']:.4f}`")
                if adf_res["is_stationary"]:
                    st.success("✔️ السلسلة (أو Log) ثابتة إحصائياً تقريباً.")
                else:
                    st.warning("❌ السلسلة غير ثابتة. تم إنشاء متغير Diff (Close_diff1) واستخدامه كميزة.")

                st.subheader("4️⃣ Feature Engineering")
                st.info(
                    "تم إضافة: SMA/EMA/RSI/MACD + Log + Diff + Lags + Returns + Volatility "
                    "ثم إسقاط الصفوف الناقصة من rolling/diff."
                )
                st.write(f"Final feature rows used for modeling: `{len(df_feat)}`")

                st.subheader("5️⃣ Model Training & Evaluation")
                all_metrics = []
                for h, metrics in results_by_h.items():
                    df_m = metrics.copy()
                    df_m.insert(0, "Horizon", f"{h} days")
                    all_metrics.append(df_m)

                all_metrics_df = pd.concat(all_metrics, ignore_index=True)
                st.dataframe(all_metrics_df, use_container_width=True)

            # ============================
            # TAB 2 – Forecasts
            # ============================
            with tab_forecasts:
                st.subheader("Future Price Forecasts (Top 3 Models)")
                st.markdown(f"**Current Price:** {current_price:.2f}")

                for h in horizons:
                    if h not in forecasts:
                        continue

                    st.markdown(f"## 🕒 {h} Days Forecast")

                    for model_name, info in forecasts[h].items():
                        col1, col2, col3 = st.columns([2, 2, 2])

                        with col1:
                            st.markdown(f"**{model_name}**")
                            st.metric(
                                "Forecast",
                                f"{info['forecast']:.2f}",
                                f"{info['diff_pct']:.2f}%"
                            )

                        with col2:
                            st.write(f"Range: {info['low']:.2f} – {info['high']:.2f}")

                        with col3:
                            if info["sentiment"] == "Bullish":
                                st.success("Bullish")
                            elif info["sentiment"] == "Bearish":
                                st.error("Bearish")
                            else:
                                st.info("Neutral")

                    st.markdown("---")

            # ============================
            # TAB 3 – Model Performance
            # ============================
            with tab_models:
                st.subheader(f"Top Models – {primary_h} Days")
                st.dataframe(primary_df, use_container_width=True)

                st.markdown("### Directional Accuracy")
                st.bar_chart(primary_df.set_index("Model")[["Directional_Accuracy"]])

                st.markdown("### RMSE")
                st.bar_chart(primary_df.set_index("Model")[["RMSE"]])

                # عرض مقارنة التوقعات مقابل القيم الفعلية لأفضل نموذج
                best_model_name = primary_df.iloc[0]["Model"]
                ds_primary = build_dataset(df_feat, feature_cols, primary_h)
                if ds_primary is not None:
                    X_test = ds_primary["X_test"]
                    y_test = ds_primary["y_test"]
                    pipe_best = models_by_h[primary_h][best_model_name]["pipeline"]
                    preds = pipe_best.predict(X_test)

                    comp_df = pd.DataFrame(
                        {
                            "Actual": y_test,
                            "Predicted": preds,
                        },
                        index=X_test.index,
                    )
                    st.markdown(f"### Predictions vs Actual – {best_model_name} ({primary_h} days)")
                    st.line_chart(comp_df)
