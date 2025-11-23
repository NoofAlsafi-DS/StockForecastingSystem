# app.py
# Advanced Stock Price Forecaster
# Yahoo Finance + Feature Engineering + Log-Return Targets + Heuristic Normality & Stationarity + Plots

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
    """يجلب السعر الحالي + البيانات التاريخية من Yahoo Finance."""
    stock = yf.Ticker(ticker)

    # سعر لحظي
    info = getattr(stock, "fast_info", {}) or {}
    current_price = info.get("last_price", None)
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
    """مؤشرات فنية: SMA, EMA, RSI, MACD."""
    df = df.copy()

    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # RSI تقريبي
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """إضافة Lags + Returns + Volatility."""
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


def check_normality_heuristic(series: pd.Series):
    """فحص تقريب للتوزيع الطبيعي باستخدام skew & kurtosis."""
    s = series.dropna()
    if len(s) == 0:
        return {"skew": np.nan, "kurtosis": np.nan, "is_normal_like": False}

    skew = float(s.skew())
    kurt = float(s.kurtosis())
    is_normal_like = (abs(skew) < 0.5) and (abs(kurt) < 1.0)

    return {
        "skew": skew,
        "kurtosis": kurt,
        "is_normal_like": is_normal_like,
    }


def check_stationarity_heuristic(series: pd.Series):
    """
    فحص تقريبي للثبات بدون ADF:
    - Autocorrelation lag1
    - تغيّر المتوسط والتباين بين 3 مقاطع.
    """
    s = series.dropna()
    if len(s) < 40:
        return {
            "autocorr_lag1": np.nan,
            "mean_range": np.nan,
            "var_range": np.nan,
            "is_stationary_like": False,
        }

    ac1 = float(s.autocorr(lag=1))

    n = len(s)
    third = n // 3
    s1 = s.iloc[:third]
    s2 = s.iloc[third:2 * third]
    s3 = s.iloc[2 * third:]

    m1, m2, m3 = s1.mean(), s2.mean(), s3.mean()
    v1, v2, v3 = s1.var(), s2.var(), s3.var()

    mean_range = float(max(m1, m2, m3) - min(m1, m2, m3))
    var_range = float(max(v1, v2, v3) - min(v1, v2, v3))

    mean_scale = abs(s.mean()) + 1e-6
    var_scale = s.var() + 1e-6

    mean_rel = mean_range / mean_scale
    var_rel = var_range / var_scale

    # تقريب: لو الارتباط عالي جدًا ومعاه تغيّر كبير في المتوسط/التباين → غير ثابتة
    is_stationary_like = not ((ac1 > 0.9) and (mean_rel > 0.3 or var_rel > 0.5))

    return {
        "autocorr_lag1": ac1,
        "mean_range": mean_range,
        "var_range": var_range,
        "is_stationary_like": is_stationary_like,
    }


def apply_log_transform(series: pd.Series):
    """تحويل Log."""
    return np.log(series)


def apply_differencing(series: pd.Series, order: int = 1):
    """Differencing من الدرجة الأولى افتراضياً."""
    return series.diff(order)


def build_dataset(df: pd.DataFrame, feature_cols, horizon: int):
    """
    تجهيز بيانات لتوقع log-return خلال horizon يوم بدل السعر نفسه.
    """
    df2 = df.copy()

    # log price
    df2["log_close_base"] = np.log(df2["Close"])

    target_col = f"target_{horizon}"
    # log-return بعد horizon يوم
    df2[target_col] = df2["log_close_base"].shift(-horizon) - df2["log_close_base"]

    df2 = df2.dropna(subset=feature_cols + [target_col])

    if len(df2) < 100:
        return None

    X = df2[feature_cols]
    y = df2[target_col]          # log-return

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
        "base_close_test": X.iloc[val_end:]["Close"],  # لسهولة الحساب لاحقاً
    }


def train_models_for_horizon(dataset, feature_cols):
    """
    تدريب 3 نماذج لكل أفق زمني مع استخدام log-return كهدف،
    ثم حساب المقاييس على الأسعار (بعد التحويل).
    """
    models_def = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=3,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),
    }

    metrics_list = []
    trained = {}

    X_train = dataset["X_train"]
    y_train = dataset["y_train"]              # log-return
    X_test = dataset["X_test"]
    y_test = dataset["y_test"]                # log-return
    base_close_test = dataset["base_close_test"].values  # الأسعار الحالية

    for name, base_model in models_def.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", base_model),
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)         # log-return متنبأ

        # نحول log-return إلى أسعار
        y_test_price = base_close_test * np.exp(y_test.values)
        y_pred_price = base_close_test * np.exp(y_pred)

        mse = mean_squared_error(y_test_price, y_pred_price)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_price, y_pred_price)
        r2 = r2_score(y_test_price, y_pred_price)

        dir_real = np.sign(y_test_price - base_close_test)
        dir_pred = np.sign(y_pred_price - base_close_test)
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
    st.subheader("Analysis Steps")
    st.markdown(
        """
        • Normality: تقريب باستخدام **skew & kurtosis**.  
        • Stationarity: تقريب باستخدام **autocorrelation + تغير المتوسط / التباين**.  
        • هدف التنبؤ = **log-return** لكل أفق زمني.  
        • النتائج تُحوّل إلى أسعار لقياس الخطأ وعرض التوقعات.
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
    current_price, df_raw = load_stock_data(ticker, years_hist)

    if df_raw.empty:
        with tab_summary:
            st.error("لم يتم جلب أي بيانات من Yahoo Finance. تحققي من رمز السهم أو عدد السنوات.")
    else:
        # 1) المؤشرات والميزات
        df_feat = add_technical_indicators(df_raw)
        df_feat = add_lag_features(df_feat)

        # 2) Normality heuristic
        normal_res = check_normality_heuristic(df_feat["Close"])
        use_log = not normal_res["is_normal_like"]
        df_feat["Close_log"] = apply_log_transform(df_feat["Close"])

        # 3) Stationarity heuristic
        series_for_stationarity = df_feat["Close_log"] if use_log else df_feat["Close"]
        stat_res = check_stationarity_heuristic(series_for_stationarity)

        # diff كسِمة
        df_feat["Close_diff1"] = apply_differencing(series_for_stationarity)

        # إسقاط القيم الناقصة
        df_feat = df_feat.dropna()

        # 4) قائمة الميزات
        feature_cols = [
            "Close", "Volume",
            "SMA_20", "SMA_50",
            "EMA_10", "EMA_20",
            "RSI", "MACD", "Signal",
            "Close_log", "Close_diff1",
            "Close_lag_1", "Close_lag_2", "Close_lag_3",
            "Close_lag_5", "Close_lag_7", "Close_lag_10", "Close_lag_14",
            "Return_1", "Return_3", "Return_7",
            "Volatility_7", "Volatility_14",
        ]
        feature_cols = [c for c in feature_cols if c in df_feat.columns]

        results_by_h = {}
        models_by_h = {}
        forecasts = {}

        # 5) تدريب النماذج
        for h in horizons:
            dataset = build_dataset(df_feat, feature_cols, horizon=h)
            if dataset is None:
                continue

            trained, df_metrics = train_models_for_horizon(dataset, feature_cols)
            results_by_h[h] = df_metrics
            models_by_h[h] = trained

            forecasts[h] = {}
            for model_name in df_metrics["Model"].head(3):
                pipe = trained[model_name]["pipeline"]
                rmse = trained[model_name]["rmse"]

                X_last = df_feat[feature_cols].iloc[[-1]]
                base_last = X_last["Close"].values[0]

                # التنبؤ بالعائد اللوغاريتمي ثم تحويله لسعر
                pred_return = float(pipe.predict(X_last)[0])
                pred = base_last * np.exp(pred_return)

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

                st.subheader("2️⃣ Normality Check (Heuristic)")
                st.write(f"Skew: `{normal_res['skew']:.4f}`, Kurtosis: `{normal_res['kurtosis']:.4f}`")
                if normal_res["is_normal_like"]:
                    st.success("✔️ السلسلة تبدو قريبة من التوزيع الطبيعي (Log مجرد سِمة إضافية).")
                else:
                    st.warning("❌ السلسلة بعيدة عن التوزيع الطبيعي. تم إنشاء Close_log واستخدامه كسِمة تساعد النموذج.")

                st.subheader("3️⃣ Stationarity Check (Heuristic)")
                st.write(f"Autocorr (lag 1): `{stat_res['autocorr_lag1']:.4f}`")
                st.write(f"Mean range (3 segments): `{stat_res['mean_range']:.4f}`")
                st.write(f"Var range (3 segments): `{stat_res['var_range']:.4f}`")
                if stat_res["is_stationary_like"]:
                    st.success("✔️ السلسلة تبدو شبه ثابتة (مع استخدام diff كميزة للتغير).")
                else:
                    st.warning("❌ السلسلة فيها اتجاه قوي / عدم ثبات. تم استخدام diff كميزة مهمة للنماذج.")

                st.subheader("4️⃣ Feature Engineering")
                st.info(
                    "تم إضافة SMA/EMA/RSI/MACD + Log + Diff + Lags + Returns + Volatility "
                    "مع حذف الصفوف المتأثرة بـ rolling/diff."
                )
                st.write(f"Final feature rows: `{len(df_feat)}`")

                # 5️⃣ توزيع البيانات قبل/بعد المعالجة
                st.subheader("5️⃣ Data Distribution Before & After Processing")

                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown("**Raw Close Price Distribution**")
                    fig1, ax1 = plt.subplots(figsize=(4, 3))
                    ax1.hist(df_raw["Close"].dropna(), bins=30, alpha=0.8)
                    ax1.set_xlabel("Close")
                    ax1.set_ylabel("Frequency")
                    ax1.set_title("Raw Close")
                    st.pyplot(fig1)

                with col_b:
                    if use_log:
                        st.markdown("**Processed (Log Close) Distribution**")
                        processed_series = df_feat["Close_log"]
                    else:
                        st.markdown("**Processed (Diff) Distribution**")
                        processed_series = df_feat["Close_diff1"]

                    fig2, ax2 = plt.subplots(figsize=(4, 3))
                    ax2.hist(processed_series.dropna(), bins=30, alpha=0.8)
                    ax2.set_xlabel("Value")
                    ax2.set_ylabel("Frequency")
                    ax2.set_title("Processed Series")
                    st.pyplot(fig2)

                st.subheader("6️⃣ Model Training & Evaluation")
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

                # بطاقات التوقع لكل أفق
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
                                f"{info['diff_pct']:.2f}%",
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

                # رسم السلسلة الزمنية + نقاط التنبؤ (Ensemble)
                st.subheader("📈 Forecast Plot (History + Ensemble Points)")

                hist_series = df_raw["Close"].copy().tail(120)
                if len(hist_series) > 0:
                    last_date = hist_series.index[-1]

                    future_points = {}
                    for h in horizons:
                        if h not in forecasts or len(forecasts[h]) == 0:
                            continue
                        values = [info["forecast"] for info in forecasts[h].values()]
                        ens_pred = float(np.mean(values))
                        future_date = last_date + pd.Timedelta(days=h)
                        future_points[future_date] = ens_pred

                    if future_points:
                        future_series = pd.Series(future_points, name="Forecast")
                        df_plot = pd.DataFrame(index=hist_series.index.union(future_series.index))
                        df_plot["Close"] = hist_series
                        df_plot["Forecast"] = future_series
                        st.line_chart(df_plot)
                    else:
                        st.info("لا توجد نقاط Forecast كافية لرسمها.")
                else:
                    st.info("البيانات التاريخية قليلة جداً لرسم السلسلة الزمنية.")

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

                best_model_name = primary_df.iloc[0]["Model"]
                ds_primary = build_dataset(df_feat, feature_cols, primary_h)
                if ds_primary is not None:
                    X_test = ds_primary["X_test"]
                    y_test = ds_primary["y_test"]
                    base_close_test = ds_primary["base_close_test"].values
                    pipe_best = models_by_h[primary_h][best_model_name]["pipeline"]
                    y_pred = pipe_best.predict(X_test)

                    y_test_price = base_close_test * np.exp(y_test.values)
                    y_pred_price = base_close_test * np.exp(y_pred)

                    comp_df = pd.DataFrame(
                        {"Actual": y_test_price, "Predicted": y_pred_price},
                        index=X_test.index,
                    )
                    st.markdown(f"### Predictions vs Actual – {best_model_name} ({primary_h} days)")
                    st.line_chart(comp_df)
