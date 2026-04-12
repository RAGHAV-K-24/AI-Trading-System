# ==========================================
# AI TRADING SYSTEM 
# ==========================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
import os

st.set_page_config(page_title="AI Trading Dashboard", layout="wide")
st.title("🚀 Real-time AI Financial Decision System")

# ==============================
# STOCK LISTS
# ==============================
india = [
"RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS","SBIN.NS",
"LT.NS","ITC.NS","HINDUNILVR.NS","KOTAKBANK.NS","AXISBANK.NS",
"BHARTIARTL.NS","ASIANPAINT.NS","MARUTI.NS","SUNPHARMA.NS","TITAN.NS"
]

us = [
"AAPL","MSFT","GOOGL","AMZN","TSLA","NVDA","META","NFLX",
"AMD","INTC","ORCL","IBM"
]

crypto = [
"BTC-USD","ETH-USD","BNB-USD","XRP-USD","ADA-USD","SOL-USD"
]

# Sidebar
category = st.sidebar.selectbox("Market", ["India","US","Crypto"])
section = st.sidebar.selectbox("Navigation", ["Dashboard","Portfolio","Comparison"])

stocks = india if category=="India" else us if category=="US" else crypto

# ==============================
# DASHBOARD
# ==============================
if section == "Dashboard":

    stock = st.selectbox("Select Asset", stocks)
    with st.spinner("📊 Loading market data..."):
       data = yf.download(stock, period="1y")
    if isinstance(data.columns, pd.MultiIndex):
       data.columns = data.columns.get_level_values(0)

    if data.empty:
        st.error("No data")
        st.stop()

    # Indicators
    data['MA20'] = data['Close'].rolling(20).mean()
    data['MA50'] = data['Close'].rolling(50).mean()

    delta = data['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100/(1+rs))

    # Chart
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name="Price",
    increasing_line_color='green',
    decreasing_line_color='red'
    ))

    fig.add_trace(go.Scatter(
    x=data.index,
    y=data['MA20'],
    name='MA20'
    ))

    fig.add_trace(go.Scatter(
    x=data.index,
    y=data['MA50'],
    name='MA50'
    ))

    fig.update_layout(
    template="plotly_dark",
    xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)
    # ML
    df = data.dropna().copy()
    df['Target'] = df['Close'].shift(-1)
    df = df.dropna()

    X = df[['Open','High','Low','Close','Volume']]
    y = df['Target']

    model = RandomForestRegressor()
    model.fit(X,y)

    prediction = float(model.predict([X.iloc[-1]])[0])
    last_close = float(df['Close'].iloc[-1])
    rsi = float(df['RSI'].iloc[-1])

    # Metrics UI
    col1, col2, col3 = st.columns(3)

    col1.metric("Current Price", round(last_close,2))
    col2.metric("Predicted Price", round(prediction,2))
    col3.metric("RSI", round(rsi,2))

    # Recommendation
    score = 0
    if prediction > last_close: score += 1
    if rsi < 30: score += 1
    if rsi > 70: score -= 1

    rec = ["SELL 📉","HOLD ⏸️","BUY 📈","STRONG BUY 🚀"][score+1]
    st.subheader(f"Recommendation: {rec}")

    # Risk
    returns = df['Close'].pct_change().dropna()
    vol = float(returns.std()*np.sqrt(252))
    sharpe = float((returns.mean()*252)/vol) if vol!=0 else 0

    st.write(f"Volatility: {round(vol,4)} | Sharpe: {round(sharpe,4)}")

    # NEWS
    st.subheader("📰 Market News")

    import os
    api_key = os.getenv("NEWS_API_KEY")
    newsapi = NewsApiClient(api_key=api_key)
    analyzer = SentimentIntensityAnalyzer()

    try:
        query = stock.split('.')[0] + " stock market"
        news = newsapi.get_everything(q=query, page_size=5)

        for article in news["articles"]:
            title = article["title"]
            score = analyzer.polarity_scores(title)['compound']

            if score > 0:
                st.success(title)
            elif score < 0:
                st.error(title)
            else:
                st.info(title)

    except:
        st.warning("News API error")
# ==============================
# PORTFOLIO 
# ==============================
if section == "Portfolio":

    import sqlite3

    conn = sqlite3.connect("portfolio.db", check_same_thread=False)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS portfolio (
        stock TEXT,
        qty INTEGER,
        buy REAL,
        date TEXT
    )
    """)
    conn.commit()

    st.markdown("## 📊 Portfolio Tracker")

    # ==============================
    # USD → INR RATE
    # ==============================
    @st.cache_data(ttl=3600)
    def get_usd_inr():
        data = yf.download("INR=X", period="1d")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        try:
            return float(data['Close'].iloc[-1])
        except:
            return 83

    usd_to_inr = get_usd_inr()

    # ==============================
    # INPUT
    # ==============================
    col1, col2, col3 = st.columns(3)

    with col1:
        p_stock = st.selectbox("Stock", stocks, key="add_stock")

    with col2:
        qty = st.number_input("Quantity", min_value=1, key="qty_input")

    with col3:
        date = st.date_input("Purchase Date", key="purchase_date")

    # ==============================
    # PRICE
    # ==============================
    @st.cache_data(ttl=60)
    def get_price(symbol):
        data = yf.download(symbol, period="1d")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        try:
            return float(data['Close'].iloc[-1])
        except:
            return 0

    with st.spinner("🔄 Fetching latest price..."):
        buy = get_price(p_stock)

    if ".NS" not in p_stock:
        buy *= usd_to_inr

    st.write(f"Buy Price (₹): {round(buy,2)}")

    # ==============================
    # ADD
    # ==============================
    if st.button("➕ Add to Portfolio", key="add_btn"):
        cursor.execute(
            "INSERT INTO portfolio VALUES (?, ?, ?, ?)",
            (p_stock, qty, buy, str(date))
        )
        conn.commit()
        st.success("Added Successfully ✅")

    # ==============================
    # LOAD FROM DB
    # ==============================
    portfolio = pd.read_sql("SELECT * FROM portfolio", conn)

    # ==============================
    # DISPLAY
    # ==============================
    if not portfolio.empty:

        results = []
        history = None

        for _, row in portfolio.iterrows():

            d = yf.download(row["stock"], period="6mo")

            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)

            if d.empty:
                continue

            price = float(d['Close'].iloc[-1])

            if ".NS" not in row["stock"]:
                price *= usd_to_inr

            value = price * row["qty"]
            invest = row["buy"] * row["qty"]
            profit = value - invest
            pct = (profit / invest) * 100 if invest != 0 else 0

            results.append({
                "Stock": row["stock"],
                "Qty": row["qty"],
                "Buy (₹)": row["buy"],
                "Current (₹)": price,
                "Investment (₹)": invest,
                "Value (₹)": value,
                "Profit (₹)": profit,
                "Return %": pct,
                "Date": row["date"]
            })

            # Growth
            hist = d['Close']

            if ".NS" not in row["stock"]:
                hist *= usd_to_inr

            hist = hist * row["qty"]

            history = hist if history is None else history.add(hist, fill_value=0)

        df = pd.DataFrame(results)

        st.markdown("### 📋 Portfolio Details")
        st.dataframe(df)

        total_value = df["Value (₹)"].sum()
        total_invest = df["Investment (₹)"].sum()
        total_profit = total_value - total_invest

        st.markdown("### 💰 Portfolio Summary")

        c1, c2, c3 = st.columns(3)

        c1.metric("Investment", f"₹{round(total_invest,2)}")
        c2.metric("Value", f"₹{round(total_value,2)}")

        if total_profit > 0:
            c3.metric("Profit", f"₹{round(total_profit,2)}", "📈")
        else:
            c3.metric("Loss", f"₹{round(total_profit,2)}", "📉")

        st.markdown("### 📈 Portfolio Performance")

        if history is not None:
            st.line_chart(history)

        st.markdown("### 📊 Allocation")
        fig = px.pie(df, names="Stock", values="Value (₹)")
        st.plotly_chart(fig, use_container_width=True)

        # DELETE
        st.markdown("### 🗑️ Remove Stock")

        remove_stock = st.selectbox("Select stock", portfolio["stock"], key="remove_stock")

        if st.button("Delete", key="delete_btn"):
            cursor.execute("DELETE FROM portfolio WHERE stock = ?", (remove_stock,))
            conn.commit()
            st.warning("Stock removed. Refresh app.")

    else:
        st.info(" portfolio.")
# ==============================
# COMPARISON 
# ==============================
if section == "Comparison":

    st.markdown("## 📊 Stock Comparison")

    selected = st.multiselect("Select Stocks", stocks, key="compare_stocks")

    if selected:

        comp = pd.DataFrame()
        returns_data = pd.DataFrame()

        with st.spinner("📈 Comparing stocks..."):

            for s in selected:

                d = yf.download(s, period="6mo")

                # Fix MultiIndex
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.get_level_values(0)

                if d.empty:
                    continue

                close = d['Close']

                # Normalize (for comparison chart)
                comp[s] = close / close.iloc[0]

                # Daily returns
                returns_data[s] = close.pct_change()

        # ==============================
        # CHART
        # ==============================
        st.markdown("### 📈 Price Comparison (Normalized)")

        if not comp.empty:
            st.line_chart(comp)

        # ==============================
        # VOLATILITY / RETURNS
        # ==============================
        st.markdown("### 📊 Risk & Return")

        stats = pd.DataFrame({
            "Return %": returns_data.mean() * 100,
            "Volatility %": returns_data.std() * 100
        })

        st.dataframe(stats)

        # ==============================
        # 📈 PRICE COMPARISON
        # ==============================
        st.markdown("### 📈  Price Comparison")
        st.line_chart(comp)

        # ==============================
        # 📊 PERFORMANCE METRICS
        # ==============================
        st.markdown("### 📊 Performance Metrics")

        results = {}

        for s in comp.columns:
            total_return = (comp[s].iloc[-1] - 1) * 100

            vol = returns_data[s].std() * np.sqrt(252)

            sharpe = (returns_data[s].mean() * 252) / vol if vol != 0 else 0

            results[s] = {
                "Return %": total_return,
                "Volatility": vol,
                "Sharpe": sharpe
            }

        metrics_df = pd.DataFrame(results).T

        st.dataframe(metrics_df.style.format({
            "Return %": "{:.2f}",
            "Volatility": "{:.4f}",
            "Sharpe": "{:.2f}"
        }))

        # ==============================
        # 🏆 BEST STOCK
        # ==============================
        best = metrics_df["Return %"].idxmax()
        st.success(f"🏆 Best Performer: {best}")

        # ==============================
        # ⚠️ RISK LEVEL
        # ==============================
        st.markdown("### ⚠️ Risk Insight")

        for s in metrics_df.index:
            vol = metrics_df.loc[s, "Volatility"]

            if vol < 0.2:
                st.info(f"{s}: Low Risk")
            elif vol < 0.4:
                st.warning(f"{s}: Medium Risk")
            else:
                st.error(f"{s}: High Risk")

        # ==============================
        # 🔥 CORRELATION HEATMAP
        # ==============================
        st.markdown("### 🔥 Correlation Matrix")

        corr = returns_data.corr()

        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu",
            title="Stock Correlation"
        )

        st.plotly_chart(fig, use_container_width=True)

# ==============================
st.caption("AI-based decision support system. Not financial advice.")
