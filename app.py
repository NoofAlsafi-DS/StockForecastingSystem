# app.py
# Advanced Stock Price Forecaster (Realtime Yahoo Finance)

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

# إعداد صفحة ستريمليت
st.set_page_config(
    page_title="Advanced Stock Price Forecaster",
    layout="wide"
)

# --------------------------- #
# 1) جلب البيانات من Yahoo    #
# --------------------------- #

def load_stock_data(ticker: str, years: int):
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

    # بيانات 1–10 سنوات
    df = stock.history(period=f"{years}y")
    df = df.dropna()

    return current_price, df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
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


def build_dataset(df: pd.DataFrame, feature_cols, horizon: int):
    df2 = df.copy()
    target_col = f"target_{horizon}"

    df2[target_col] = df2["Close"].shift(-horizon)
    df2 = df2.dropna(subset=feature_cols + [target_col])

    if len(df2) < 50:
        return None

    X = df2[feature_cols]
    y = df2[target_col]

    n = len(df2)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    return {
        "X_train": X.iloc[:train_end],
        "y_train": y.iloc[:train_end],
        "X_val": X.iloc[train_end:val_end],
        "y_val": y.iloc[train_end:val_end],
        "X_test": X.iloc[val_end:],
        "y_test": y.iloc[val_end:],
    }


def train_models_for_horizon(dataset, feature_cols):
    models_def = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
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

        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

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

    metrics_df = pd.DataFrame(metrics_list).sort_values("Directional_Accuracy", ascending=False)
    return trained, metrics_df

# --------------------------- #
# 2) الإعدادات – Sidebar      #
# --------------------------- #

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
    horizons_selected = st.multiselect(
        "Forecast Horizons", horizons_labels, default=horizons_labels
    )

    horizon_map = {"7 Days": 7, "14 Days": 14, "30 Days": 30}
    horizons = [horizon_map[h] for h in horizons_selected]

    train_button = st.button("🚀 Train Models & Forecast")

# --------------------------- #
# 3) Tabs                     #
# --------------------------- #

st.title("📈 Advanced Stock Price Forecaster")

tab_summary, tab_forecasts, tab_models = st.tabs(
    ["Pipeline Summary", "Future Price Forecasts", "Model Performance"]
)

if not train_button:
    with tab_summary:
        st.info("⬅️ اختر الإعدادات من القائمة ثم اضغط **Train Models & Forecast**.")
    with tab_forecasts:
        st.warning("لم يتم تدريب النماذج بعد.")
    with tab_models:
        st.warning("لم يتم تدريب النماذج بعد.")
else:
    current_price, df_raw = load_stock_data(ticker, years_hist)
    df_feat = add_technical_indicators(df_raw).dropna()

    feature_cols = [
        "Close", "Volume", "SMA_20", "SMA_50",
        "EMA_10", "EMA_20", "RSI", "MACD", "Signal"
    ]

    results_by_h = {}
    models_by_h = {}
    forecasts = {}

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
            pred = float(pipe.predict(X_last)[0])

            diff_pct = (pred - current_price) / current_price * 100
            low = pred - rmse
            high = pred + rmse

            sentiment = "Bullish" if diff_pct > 1 else "Bearish" if diff_pct < -1 else "Neutral"

            forecasts[h][model_name] = {
                "forecast": pred,
                "diff_pct": diff_pct,
                "low": low,
                "high": high,
                "sentiment": sentiment
            }

    # ---------------------- TAB 1 ---------------------- #
    with tab_summary:
        st.subheader("Data Collection")
        st.success(f"Loaded {len(df_raw)} rows for **{ticker}**. Current Price: **{current_price:.2f}**")

        st.subheader("Feature Engineering")
        st.info("Added SMA, EMA, RSI, MACD, and other indicators.")

        st.subheader("Model Training")
        st.success("All models trained successfully ✔️")

        all_metrics = []
        for h, metrics in results_by_h.items():
            df_m = metrics.copy()
            df_m.insert(0, "Horizon", f"{h} days")
            all_metrics.append(df_m)

        st.dataframe(pd.concat(all_metrics), use_container_width=True)

    # ---------------------- TAB 2 ---------------------- #
    with tab_forecasts:
        st.subheader("Future Price Forecasts")
        st.markdown(f"**Current Price:** {current_price:.2f}")

        for h in horizons:
            if h not in forecasts:
                continue

            st.markdown(f"## 🕒 {h} Days Forecast")

            for model_name, info in forecasts[h].items():
                col1, col2, col3 = st.columns([2, 2, 2])

                with col1:
                    st.markdown(f"**{model_name}**")
                    st.metric("Forecast", f"{info['forecast']:.2f}", f"{info['diff_pct']:.2f}%")

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

    # ---------------------- TAB 3 ---------------------- #
    with tab_models:
        primary_h = horizons[0]
        primary_df = results_by_h[primary_h]

        st.subheader(f"Top Models – {primary_h} Days")
        st.dataframe(primary_df, use_container_width=True)

        st.markdown("### Directional Accuracy")
        st.bar_chart(primary_df.set_index("Model")[["Directional_Accuracy"]])

        st.markdown("### RMSE")
        st.bar_chart(primary_df.set_index("Model")[["RMSE"]])
