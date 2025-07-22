# 🌈 Spectra — Autonomous AI Trading Bot Powered by FastAI + Recall

**Spectra** is a fully autonomous, AI-powered crypto trading bot built for the **Autonomous Apes AI Trading Competition** hosted by the **Recall Network**. It uses deep learning to predict fair token prices, monitors real-time market conditions through Recall's API, and executes trades across a multi-token portfolio — all autonomously, safely, and with reasoning.

---

## 🧠 What is Spectra?

Spectra is designed to:
- 📡 Fetch live prices using the **Recall API**
- 🧠 Predict fair token prices using a **FastAI-trained neural network**
- 🔄 Make autonomous **buy/sell/hold** decisions across a **4-token portfolio**
- 🤝 Execute trades via the Recall Trading API with embedded explanations

---

## 📈 Core Strategy

Spectra follows a transparent, rule-based trading strategy:

| Action | Trigger Condition                          |
|--------|---------------------------------------------|
| BUY    | Market Price < 90% of Predicted Price      |
| SELL   | Market Price > 110% of Predicted Price     |
| HOLD   | Price within ±10% of Predicted Fair Value  |

This logic enables Spectra to act decisively on clear underpricing or overvaluation signals, while avoiding noisy or uncertain trades.

---

## ⚙️ Model Architecture

- Framework: `fastai`
- Type: Tabular Neural Network
- Input Features:
  - 24h Volume
  - Circulating Supply
  - Total Supply
  - Market Cap
- Output Target: Fair Market Price (USD)
- Exported Model: `nn_model.pkl`
- Scaler: `scaler.pkl` using `scikit-learn` StandardScaler
- Performance: Low RMSE and strong generalization on validation set

---

## 💼 Portfolio Tokens

Spectra supports a 4-token diversified portfolio:

| Token Name     | Symbol | Recall Address                                 |
|----------------|--------|-------------------------------------------------|
| Alien Worlds   | TLM    | `0x5e4e65926ba27467555eb562121fac00d24e9dd2`   |
| Wrapped Ether  | WETH   | `0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2`   |
| Wrapped BTC    | WBTC   | `0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599`   |
| Chainlink      | LINK   | `0x514910771AF9Ca656af840dff83E8264EcF986CA`   |

---

## 🧠 Example Predictions


---

## 🧪 Risk Controls

- ❌ Skips trades on missing predictions or prices
- 💰 Max trade cap: `$100` per token per decision cycle
- 🛡️ Slippage protection: 0.5%
- 🧾 Trade reasoning included in each request for transparency

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```
git clone https://github.com/ofemeteng/spectra.git
cd spectra```

### 2. Install dependencies
pip install -r requirements.txt

fastai
pandas
scikit-learn
requests
python-dotenv

### Set Recall API Key
RECALL_API_KEY=your_recall_api_key


### Run AI Trading Agent
python ai_trading_agent.py

## Contact 

GitHub: @ofemeteng

Twitter: @ofemetengE



