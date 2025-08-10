import streamlit as st
import pandas as pd
import yfinance as yf
import os
from openai import OpenAI
from dotenv import load_dotenv
import json
import re

load_dotenv()
client = OpenAI()


# Simple hardcoded users for MVP
USERS = {
    "amrao": "password123",
}

def login():
    st.title("TradeHelp Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USERS and USERS[username] == password:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
        else:
            st.error("Invalid username or password")

def fetch_prices(tickers):
    data = {}
    for t in tickers:
        try:
            ticker = yf.Ticker(t)
            price = ticker.history(period="1d")["Close"].iloc[-1]
            data[t] = price
        except Exception as e:
            data[t] = None
    return data

def build_prompt(portfolio, prices):
    prompt = """You are a stock trading assistant. For each stock, given the quantity, buy price, and current price, recommend one action: BUY, SELL, or HOLD. Also provide a brief reasoning (max 2 sentences).

Respond in JSON with a list of objects like:
[{"symbol": "...", "action": "BUY|SELL|HOLD", "reason": "..."}]

Portfolio:
"""
    for row in portfolio.itertuples():
        symbol = row.Ticker.upper()
        qty = row.Quantity
        buy_price = row.BuyPrice
        current_price = prices.get(symbol, None)
        prompt += f"{symbol}: qty={qty}, buy_price={buy_price}, current_price={current_price}\n"

    prompt += "\nOnly respond with JSON."

    return prompt

def get_recommendations(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=500,
        )
        text = response.choices[0].message.content
        return text
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return None

import json
import re

def main_app():
    st.title(f"Welcome {st.session_state['username']} to TradeHelp")
    st.write("Upload your portfolio CSV with columns: Ticker, Quantity, BuyPrice")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        portfolio = pd.read_csv(uploaded_file)
        if all(col in portfolio.columns for col in ["Ticker", "Quantity", "BuyPrice"]):
            tickers = portfolio["Ticker"].str.upper().unique().tolist()
            prices = fetch_prices(tickers)
            st.write("Current Prices:", prices)
            prompt = build_prompt(portfolio, prices)
            st.write("Prompt sent to LLM:")
            st.code(prompt)
            result_text = get_recommendations(prompt)
            if result_text:
                # Strip whitespace
                cleaned = result_text.strip()
                # Try to extract JSON array with regex in case extra text around JSON
                match = re.search(r'\[.*\]', cleaned, re.DOTALL)
                if match:
                    json_text = match.group(0)
                    try:
                        data = json.loads(json_text)
                        recommendations = pd.DataFrame(data)

                        def color_row(row):
                            if row["action"] == "BUY":
                                return ['background-color: #d4f4dd']*len(row)
                            elif row["action"] == "SELL":
                                return ['background-color: #f4d4d4']*len(row)
                            else:
                                return ['background-color: #f9f0c1']*len(row)

                        st.write("Recommendations:")
                        st.dataframe(recommendations.style.apply(color_row, axis=1))
                    except Exception as e:
                        st.error(f"Error parsing JSON data: {e}")
                        st.write("Raw extracted JSON:")
                        st.write(json_text)
                else:
                    st.error("No JSON array found in the LLM response.")
                    st.write("Raw response:")
                    st.write(result_text)
            else:
                st.error("Received empty response from LLM.")
        else:
            st.error("CSV must contain columns: Ticker, Quantity, BuyPrice")

def main():
    if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
        login()
    else:
        main_app()

if __name__ == "__main__":
    main()