import os
import json
import requests
import pandas as pd
import pickle
from dotenv import load_dotenv
from fastai.tabular.all import load_learner
from datetime import datetime

# Load environment variables
load_dotenv()
API_KEY = os.getenv("RECALL_API_KEY")
BASE_URL = "https://api.competitions.recall.network"

# Load model and scaler
learn = load_learner("nn_model.pkl")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
TRADE_AMOUNT = 100  # in USDC
BUY_THRESHOLD = 0.90
SELL_THRESHOLD = 1.10

# 4-token portfolio (update addresses as needed)
PORTFOLIO = {
    "TLM": {
        "token_address": "0x5e4e65926ba27467555eb562121fac00d24e9dd2",  # Replace with TLM address
        "features": {
            "24h_Volume": 110179781,
            "Circulating_Supply": 4413922778,
            "Total_Supply": 6449215559,
            "Market_Cap": 65768788
        }
    },
    "WETH": {
        "token_address": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        "features": {
            "24h_Volume": 3300000000,
            "Circulating_Supply": 120000000,
            "Total_Supply": 120000000,
            "Market_Cap": 350000000000
        }
    },
    "WBTC": {
        "token_address": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
        "features": {
            "24h_Volume": 450000000,
            "Circulating_Supply": 190000,
            "Total_Supply": 210000,
            "Market_Cap": 14000000000
        }
    },
    "LINK": {
        "token_address": "0x514910771AF9Ca656af840dff83E8264EcF986CA",
        "features": {
            "24h_Volume": 400000000,
            "Circulating_Supply": 560000000,
            "Total_Supply": 1000000000,
            "Market_Cap": 8000000000
        }
    }
}

def fetch_recall_price(token_address):
    url = f"{BASE_URL}/api/price"
    params = {
        "token": token_address,
        "chain": "svm",
        "specificChain": "mainnet"
    }
    headers = {"Authorization": f"Bearer {API_KEY}"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        price = float(resp.json().get("price", 0))
        return price
    except Exception as e:
        print(f"❌ Failed to fetch price for {token_address}: {e}")
        return None

def prepare_input(features):
    df = pd.DataFrame([features])
    cols = ['24h_Volume', 'Circulating_Supply', 'Total_Supply', 'Market_Cap']
    df[cols] = scaler.transform(df[cols])
    return df

def predict_price(features):
    df = prepare_input(features)
    dl = learn.dls.test_dl(df)
    pred, *_ = learn.get_preds(dl=dl)
    return pred[0].item()

def decide_action(predicted, market):
    ratio = market / predicted
    if ratio < BUY_THRESHOLD:
        return "buy"
    elif ratio > SELL_THRESHOLD:
        return "sell"
    else:
        return "hold"

def execute_trade(from_token, to_token, amount_usdc, reason):
    url = f"{BASE_URL}/api/trade/execute"
    payload = {
        "fromToken": from_token,
        "toToken": to_token,
        "amount": str(amount_usdc),
        "reason": reason,
        "slippageTolerance": "0.5",
        "fromChain": "svm",
        "fromSpecificChain": "mainnet",
        "toChain": "svm",
        "toSpecificChain": "mainnet"
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=20)
        resp.raise_for_status()
        print("✅ Trade executed:", resp.json())
    except Exception as e:
        print(f"❌ Trade failed: {e}")

def run_trading_bot():
    print(f"\n🚀 Running Recall Trading Bot @ {datetime.now()}...\n")
    for symbol, token_info in PORTFOLIO.items():
        print(f"📊 Evaluating {symbol}...")

        price = fetch_recall_price(token_info["token_address"])
        if not price:
            print(f"⚠️ Skipping {symbol} due to missing price.")
            continue

        pred = predict_price(token_info["features"])
        action = decide_action(pred, price)

        print(f"🧠 Predicted: ${pred:.5f} | 🪙 Market: ${price:.5f} | Action: {action.upper()}")

        if action == "buy":
            execute_trade(USDC, token_info["token_address"], TRADE_AMOUNT, f"BUY {symbol} @ {price:.4f}, predicted {pred:.4f}")
        elif action == "sell":
            execute_trade(token_info["token_address"], USDC, TRADE_AMOUNT, f"SELL {symbol} @ {price:.4f}, predicted {pred:.4f}")
        else:
            print("✅ Holding position.\n")

if __name__ == "__main__":
    run_trading_bot()
