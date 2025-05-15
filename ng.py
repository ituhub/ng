import os
os.makedirs("model", exist_ok=True)

import pandas as pd
import numpy as np
import requests
import streamlit as st
import matplotlib.pyplot as plt

# Sklearn
from sklearn.preprocessing import MinMaxScaler

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer  # REMOVED TRANSFORMER: Removed MultiHeadAttention import

# PyTorch, PyTorch Geometric
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch_geometric.data import Data as GeoData
    from torch_geometric.nn import GCNConv
except ImportError:
    st.warning("torch_geometric not installed. GNN features disabled.")
    GeoData = None

# Stable-Baselines3 for DQN
try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.vec_env import DummyVecEnv
    import gym
    from gym import spaces
except ImportError:
    st.warning("stable-baselines3/gym not installed. DQN features disabled.")
    DQN = None

# hmmlearn for Hidden Markov Model
try:
    import joblib
    from hmmlearn import hmm
except ImportError:
    st.warning("hmmlearn not installed. HMM features disabled.")
    hmm = None

# torchdiffeq for Neural ODE
try:
    from torchdiffeq import odeint
except ImportError:
    st.warning("torchdiffeq not installed. Neural ODE features disabled.")
    odeint = None

# Optional: sentiment / google trends
try:
    from transformers import pipeline
except ImportError:
    pipeline = None

try:
    from pytrends.request import TrendReq
except ImportError:
    TrendReq = None


###############################################################################
# 1) Data fetching
###############################################################################
def fetch_live_data(tickers, retries=3):
    data = {}
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        st.error("Missing FMP_API_KEY env variable.")
        return data

    for t in tickers:
        for attempt in range(retries):
            try:
                t_api = t.replace('/', '')
                url = f'https://financialmodelingprep.com/api/v3/historical-chart/15min/{t_api}?apikey={api_key}'
                r = requests.get(url)
                r.raise_for_status()
                jdata = r.json()
                if not jdata:
                    st.warning(f"No data for {t}")
                    continue
                df = pd.DataFrame(jdata)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.rename(
                    columns={'open':'Open','close':'Close','high':'High','low':'Low','volume':'Volume'},
                    inplace=True
                )
                df.sort_index(inplace=True)
                data[t] = df
                break
            except Exception as e:
                if attempt < retries-1:
                    st.warning(f"Retry fetch {t}, attempt {attempt+1}")
                else:
                    st.error(f"Failed to fetch {t}: {e}")
        else:
            st.error(f"No data for {t} after {retries} attempts.")
    return data


###############################################################################
# 2) Basic Indicators (matching train.py)
###############################################################################
def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_MACD(series):
    exp1 = series.ewm(span=12, adjust=False).mean()
    exp2 = series.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    sig  = macd.ewm(span=9, adjust=False).mean()
    return macd, sig

def compute_stochastic_oscillator(df, k_period=14, d_period=3):
    df = df.copy()
    min_low = df["Low"].rolling(window=k_period).min()
    max_high = df["High"].rolling(window=k_period).max()
    df["%K"] = 100 * (df["Close"] - min_low) / (max_high - min_low)
    df["%D"] = df["%K"].rolling(window=d_period).mean()
    return df

def compute_bollinger_bands(df, period=20, num_std=2):
    df = df.copy()
    df["BB_Middle"] = df["Close"].rolling(window=period).mean()
    df["BB_Std"] = df["Close"].rolling(window=period).std()
    df["BB_Upper"] = df["BB_Middle"] + num_std * df["BB_Std"]
    df["BB_Lower"] = df["BB_Middle"] - num_std * df["BB_Std"]
    return df

def compute_atr(df, period=14):
    df = df.copy()
    df["H-L"]  = df["High"] - df["Low"]
    df["H-PC"] = abs(df["High"] - df["Close"].shift(1))
    df["L-PC"] = abs(df["Low"] - df["Close"].shift(1))
    df["TR"]   = df[["H-L","H-PC","L-PC"]].max(axis=1)
    df["ATR"]  = df["TR"].rolling(window=period).mean()
    return df

def compute_ema(series, span=21):
    return series.ewm(span=span, adjust=False).mean()

def compute_OBV(df):
    df = df.copy()
    df["Direction"] = np.where(df["Close"] > df["Close"].shift(1), 1, -1)
    df["Direction"] = np.where(df["Close"] == df["Close"].shift(1), 0, df["Direction"])
    df["OBV"] = (df["Volume"] * df["Direction"]).cumsum()
    return df

def compute_ADX(df, n=14):
    def compute_true_range(df_):
        pc = df_["Close"].shift(1)
        hl = df_["High"] - df_["Low"]
        hp = abs(df_["High"] - pc)
        lp = abs(df_["Low"] - pc)
        return pd.concat([hl, hp, lp], axis=1).max(axis=1)

    def compute_positive_DM(df_):
        up_move = df_["High"] - df_["High"].shift(1)
        dn_move = df_["Low"].shift(1) - df_["Low"]
        plus_dm = np.where((up_move>dn_move)&(up_move>0), up_move, 0)
        return plus_dm

    def compute_negative_DM(df_):
        up_move = df_["High"] - df_["High"].shift(1)
        dn_move = df_["Low"].shift(1) - df_["Low"]
        minus_dm = np.where((dn_move>up_move)&(dn_move>0), dn_move, 0)
        return minus_dm

    df = df.copy()
    df["TR"] = compute_true_range(df)
    df["+DM"] = compute_positive_DM(df)
    df["-DM"] = compute_negative_DM(df)

    df["TRn"] = df["TR"].rolling(window=n).sum()
    df["+DMn"] = df["+DM"].rolling(window=n).sum()
    df["-DMn"] = df["-DM"].rolling(window=n).sum()

    df["+DI"] = 100*(df["+DMn"] / df["TRn"])
    df["-DI"] = 100*(df["-DMn"] / df["TRn"])
    df["DX"] = 100*(abs(df["+DI"] - df["-DI"])) / (df["+DI"] + df["-DI"])
    df["ADX"] = df["DX"].rolling(window=n).mean()
    return df


###############################################################################
# 3) Enhanced Feature Engineering
###############################################################################
def enhance_features(df):
    df = df.copy()
    df["RSI"] = compute_RSI(df["Close"])
    df["MACD"], df["MACD_Signal"] = compute_MACD(df["Close"])
    df = compute_stochastic_oscillator(df)
    df = compute_bollinger_bands(df)
    df = compute_atr(df)
    df["EMA_21"] = compute_ema(df["Close"], span=21)
    df = compute_OBV(df)
    df = compute_ADX(df)

    df["MA_Short"] = df["Close"].rolling(5).mean()
    df["MA_Long"]  = df["Close"].rolling(20).mean()

    df["Lag_1"] = df["Close"].shift(1)
    df["Lag_2"] = df["Close"].shift(2)
    df["Lag_3"] = df["Close"].shift(3)

    df["RSI_MACD_Interaction"] = df["RSI"] * df["MACD"]
    return df


def classify_volatility(df, lookback=14):
    df = df.copy()
    if 'ATR' not in df.columns:
        df = compute_atr(df, period=lookback)
    df['RollingStd'] = df['Close'].rolling(lookback).std()
    df['VolatilityScore'] = 0.5*df['ATR'] + 0.5*df['RollingStd']

    v = df['VolatilityScore'].dropna()
    if len(v)<3:
        df['VolatilityClass'] = 'Low'
        return df
    low_t, high_t = v.quantile([0.33,0.66])
    conds = [
        df['VolatilityScore']<=low_t,
        (df['VolatilityScore']>low_t)&(df['VolatilityScore']<=high_t),
        df['VolatilityScore']>high_t
    ]
    df['VolatilityClass'] = np.select(conds, ['Low','Medium','High'], default='Low')
    return df


###############################################################################
# 4) Prepare Single-Time-Step Data to match train_cnn_lstm_attention / train_transformer_model
#    (We keep the function but won't do Transformer anymore.)
###############################################################################
def prepare_single_time_data(df):
    """
    In the original training, we used all numeric columns except "Close"
    as features, then "Close" as the y. We'll do the same at inference:
      - gather all numeric except "Close"
      - reshape => (samples, 1, features)
    We produce X, y, and a scaler to invert predictions if desired.
    """
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if "Close" not in numeric_df.columns:
        # no close => can't do a direct predicted close
        return None, None, None

    feature_cols = [c for c in numeric_df.columns if c != "Close"]
    X_data = numeric_df[feature_cols].values  # shape (samples, features)
    y_data = numeric_df["Close"].values       # shape (samples,)

    scaler = MinMaxScaler()
    X_data_scaled = scaler.fit_transform(X_data)

    # Now reshape => (samples, 1, features)
    X_reshaped = X_data_scaled.reshape((X_data.shape[0], 1, X_data.shape[1]))
    return X_reshaped, y_data, scaler, feature_cols


###############################################################################
# 5) Our Keras AttentionLayer
###############################################################################
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        feat_dim = input_shape[-1]
        self.W = self.add_weight("att_weight", shape=(feat_dim, feat_dim), initializer="normal")
        self.b = self.add_weight("att_bias", shape=(feat_dim,), initializer="zeros")
        self.u = self.add_weight("att_u", shape=(feat_dim, 1), initializer="normal")
        super().build(input_shape)

    def call(self, inputs):
        # (batch, time, features)
        x_tanh = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=[2,0]) + self.b)
        score = tf.tensordot(x_tanh, self.u, axes=[2,0])  # => (batch, time, 1)
        att_weights = tf.nn.softmax(score, axis=1)
        context_vector = att_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)  # => (batch, features)
        return context_vector


###############################################################################
# REMOVED TRANSFORMER: The entire load_transformer_model function was removed
###############################################################################


###############################################################################
# 6) Load CNN+LSTM + Attention
###############################################################################
def load_cnn_lstm_attention_model(ticker):
    path = f"model/{ticker}_cnn_lstm_attention_model_tuned.h5"
    if not os.path.exists(path):
        raise FileNotFoundError(f"No CNN+LSTM model at {path}")
    return load_model(path, custom_objects={"AttentionLayer": AttentionLayer})


###############################################################################
# 7) GNN, DQN, HMM, Neural ODE
###############################################################################
class GNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, 32)
        self.fc1   = nn.Linear(32, 16)
        self.fc2   = nn.Linear(16, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x.squeeze()

def load_gnn_model(num_features):
    model = GNN(num_features)
    path = "model/gnn_model.pth"
    if not os.path.exists(path):
        raise FileNotFoundError(f"No GNN model at {path}")
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_with_gnn(df):
    if GeoData is None:
        st.warning("PyTorch Geometric not installed, skipping GNN.")
        return pd.Series([], index=df.index)

    df2 = df.reset_index(drop=True).copy()
    numeric_df = df2.select_dtypes(include=[np.number])
    if 'Close' not in numeric_df.columns:
        st.warning("No 'Close' in data, skipping GNN.")
        return pd.Series([], index=df.index)

    feats = numeric_df.drop(columns=['Close']).values
    edge_index = torch.tensor(
        [list(range(len(df2)-1)), list(range(1, len(df2)))],
        dtype=torch.long
    )
    x = torch.tensor(feats, dtype=torch.float)

    model = load_gnn_model(x.shape[1])
    with torch.no_grad():
        out = model(x, edge_index)
    pred_close = pd.Series(out.numpy(), index=df2.index)
    pred_close.name = "GNN_Predicted_Close"
    pred_close.index = df.index
    return pred_close


###############################################################################
# DQN
###############################################################################
class TradingEnv(gym.Env):
    metadata = {'render.modes':['human']}
    def __init__(self, df, scaler, initial_balance=10000):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.scaler = scaler
        self.action_space = spaces.Discrete(3)
        num_features = self.df.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(num_features,), dtype=np.float32)
        self.current_step = 0
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.num_shares = 0
        self.net_worth = initial_balance
        self.max_steps = len(self.df)-1

    def _next_observation(self):
        obs_row = self.df.iloc[self.current_step].values
        obs_scaled = self.scaler.transform([obs_row])[0]
        return obs_scaled.astype(np.float32)

    def step(self, action):
        self.current_step += 1
        done = (self.current_step >= self.max_steps)
        reward = 0

        price = self.df.loc[self.current_step, "Close"]
        if action == 0:  # Buy
            max_shares = self.balance // price
            if max_shares>0:
                self.num_shares += max_shares
                self.balance -= max_shares * price
        elif action == 1: # Sell
            if self.num_shares>0:
                self.balance += self.num_shares * price
                self.num_shares=0

        self.net_worth = self.balance + self.num_shares * price
        reward = self.net_worth - self.initial_balance
        return self._next_observation(), reward, done, {}

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.num_shares = 0
        return self._next_observation()

    def render(self,mode='human'):
        pass

def load_dqn_model(ticker):
    if DQN is None:
        st.warning("stable-baselines3 not installed, skipping DQN.")
        return None
    path = f"model/dqn_model_{ticker}.zip"
    if not os.path.exists(path):
        st.warning(f"No DQN model at {path}")
        return None
    return DQN.load(path)

def predict_with_dqn(df, ticker):
    if DQN is None:
        return None
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        st.warning("No numeric data for DQN environment. Skipping.")
        return None

    scaler = MinMaxScaler()
    scaler.fit(numeric_df)
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = TradingEnv(numeric_df, scaler)
    env = DummyVecEnv([lambda: env])
    model = load_dqn_model(ticker)
    if model is None:
        return None

    obs = env.reset()
    action, _states = model.predict(obs, deterministic=True)
    action_map = {0:"Buy", 1:"Sell", 2:"Hold"}
    return action_map.get(action[0], "Hold")


###############################################################################
# HMM
###############################################################################
def load_hmm_model():
    path = "model/hmm_model.pkl"
    if not os.path.exists(path):
        st.warning(f"No HMM model found at {path}")
        return None
    return joblib.load(path)

def predict_with_hmm(df):
    if hmm is None:
        st.warning("hmmlearn not installed, skipping HMM.")
        return pd.Series([], index=df.index)

    df2 = df.copy()
    df2['Returns'] = df2['Close'].pct_change().fillna(0.0)
    model = load_hmm_model()
    if model is None:
        return pd.Series([], index=df.index)

    arr = df2[['Returns']].values
    states = model.predict(arr)
    return pd.Series(states, index=df2.index, name="HMM_Regime")


###############################################################################
# Neural ODE
###############################################################################
class ODEFunc(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, t, x):
        return self.net(x)

def load_neural_ode_func():
    path = "model/neural_ode_func.pth"
    if not os.path.exists(path):
        st.warning(f"No Neural ODE function at {path}")
        return None
    func = ODEFunc(hidden_dim=16)
    func.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    func.eval()
    return func

def predict_with_neural_ode(df):
    if odeint is None:
        return pd.Series([], index=df.index)
    if 'Close' not in df.columns:
        return pd.Series([], index=df.index)

    close_prices = df['Close'].values.astype(np.float32)
    N = len(close_prices)
    if N < 2:
        return pd.Series([], index=df.index)

    t = torch.linspace(0, 1, steps=N)
    x0 = torch.tensor([close_prices[0]])
    func = load_neural_ode_func()
    if func is None:
        return pd.Series([], index=df.index)

    with torch.no_grad():
        pred = odeint(func, x0, t)
    arr = pred.view(-1).numpy()
    return pd.Series(arr, index=df.index, name="NeuralODE_Close")


###############################################################################
# REMOVED TRANSFORMER: The 'decide_trade_action' function that combined 
# transformer, CNN, ODE, etc. is optional. If you don't need it, remove it.
# Otherwise, keep it but remove any references to 'transformer_close'.
###############################################################################
def decide_trade_action(cnn_close, ode_close, dqn_action, hmm_regime, current_price):
    """
    This version no longer references Transformer predictions.
    We just rely on the CNN, Neural ODE, DQN, and HMM.
    """
    votes = {'Buy': 0.0, 'Sell': 0.0, 'Hold': 0.0}

    # CNN+LSTM vote
    if cnn_close:
        if cnn_close > current_price * 1.01:
            votes['Buy'] += 0.4
        elif cnn_close < current_price * 0.99:
            votes['Sell'] += 0.4
        else:
            votes['Hold'] += 0.4

    # Neural ODE vote
    if ode_close:
        if ode_close > current_price * 1.01:
            votes['Buy'] += 0.3
        elif ode_close < current_price * 0.99:
            votes['Sell'] += 0.3
        else:
            votes['Hold'] += 0.3

    # DQN vote (discrete)
    if dqn_action:
        if dqn_action == "Buy":
            votes['Buy'] += 0.2
        elif dqn_action == "Sell":
            votes['Sell'] += 0.2
        else:
            votes['Hold'] += 0.2

    # HMM vote (if regime 0 = bear, 1 = bull, 2 = uncertain)
    if hmm_regime == 1:
        votes['Buy'] += 0.1
    elif hmm_regime == 0:
        votes['Sell'] += 0.1
    else:
        votes['Hold'] += 0.1

    final_decision = max(votes, key=votes.get)
    return final_decision, votes

###############################################################################
# 8) Generate trade decision (CNN-based only, no Transformer)
###############################################################################
def generate_trade_decision(df, df_preds):
    try:
        current_price = df["Close"].iloc[-1]
        cnn_close = df_preds["CNN_LSTM_Predicted_Close"].iloc[-1]

        votes = {"Buy": 0, "Sell": 0, "Hold": 0}

        # CNN+LSTM vote
        if cnn_close > current_price * 1.01:
            votes["Buy"] += 1
        elif cnn_close < current_price * 0.99:
            votes["Sell"] += 1
        else:
            votes["Hold"] += 1

        # GNN vote
        if "GNN_Predicted_Close" in df_preds.columns:
            gnn_close = df_preds["GNN_Predicted_Close"].iloc[-1]
            if gnn_close > current_price * 1.01:
                votes["Buy"] += 1
            elif gnn_close < current_price * 0.99:
                votes["Sell"] += 1
            else:
                votes["Hold"] += 1

        # Neural ODE vote
        if "NeuralODE_Close" in df_preds.columns:
            ode_close = df_preds["NeuralODE_Close"].iloc[-1]
            if ode_close > current_price * 1.01:
                votes["Buy"] += 1
            elif ode_close < current_price * 0.99:
                votes["Sell"] += 1
            else:
                votes["Hold"] += 1

        # HMM regime vote
        if "HMM_Regime" in df_preds.columns:
            hmm_regime = df_preds["HMM_Regime"].iloc[-1]
            if hmm_regime == 1:
                votes["Buy"] += 1
            elif hmm_regime == 0:
                votes["Sell"] += 1
            else:
                votes["Hold"] += 1

        # DQN vote
        if "DQN_Action" in df_preds.columns:
            dqn_action = df_preds["DQN_Action"].iloc[-1]
            if dqn_action in votes:
                votes[dqn_action] += 1

        final_action = max(votes, key=votes.get)

        return {
            "Current Price": f"{current_price:.2f}",
            "CNN Prediction": f"{cnn_close:.2f}",
            "Votes": votes,
            "Recommended Action": final_action,
            "Reason": f"Smart Voting Result: {final_action} (votes: {votes})"
        }

    except Exception as e:
        return {"Error": f"Smart trade decision error: {e}"}

###############################################################################
# 9) A Simple Display for CNN Predictions
###############################################################################
def display_cnn_predictions(df, df_preds, ticker):
    """
    Show CNN-LSTM + optional GNN + ODE ensemble prediction.
    All predictions are normalized to CNN scale to avoid outlier effects.
    """
    if "CNN_LSTM_Predicted_Close" not in df_preds.columns:
        st.error("No CNN-LSTM predictions found.")
        return

    current_price = df["Close"].iloc[-1]
    cnn_close = df_preds["CNN_LSTM_Predicted_Close"].iloc[-1]

    # === Normalize helper ===
    def normalize_to_cnn_scale(series_to_adjust, reference_series):
        try:
            factor = reference_series[-10:].mean() / series_to_adjust[-10:].mean()
            return series_to_adjust * factor
        except:
            return series_to_adjust

    # === Prepare model outputs ===
    votes = []
    weights = []
    weight_labels = []

    if not np.isnan(cnn_close):
        votes.append(cnn_close)
        weights.append(0.4)
        weight_labels.append("CNN")

    # GNN
    if "GNN_Predicted_Close" in df_preds.columns:
        gnn_series = df_preds["GNN_Predicted_Close"].dropna()
        if not gnn_series.empty:
            gnn_series = normalize_to_cnn_scale(gnn_series, df_preds["CNN_LSTM_Predicted_Close"])
            gnn_close = gnn_series.iloc[-1]
            votes.append(gnn_close)
            weights.append(0.2)
            weight_labels.append("GNN")

    # Neural ODE
    if "NeuralODE_Close" in df_preds.columns:
        ode_series = df_preds["NeuralODE_Close"].dropna()
        if not ode_series.empty:
            ode_series = normalize_to_cnn_scale(ode_series, df_preds["CNN_LSTM_Predicted_Close"])
            ode_close = ode_series.iloc[-1]
            votes.append(ode_close)
            weights.append(0.3)
            weight_labels.append("ODE")

    if votes:
        total_weight = sum(weights)
        combined_close = sum(v * w for v, w in zip(votes, weights)) / total_weight
        weight_ratio = " + ".join([f"{int(w*100)}% {label}" for w, label in zip(weights, weight_labels)])
    else:
        combined_close = cnn_close
        weight_ratio = "100% CNN only"

    # === Confidence and action ===
    stop_loss = current_price * 0.98
    take_profit = current_price * 1.02

    N = min(10, len(df_preds))
    real_vals = df["Close"].iloc[-N:].values
    pred_vals = df_preds["CNN_LSTM_Predicted_Close"].iloc[-N:].values
    mape = np.mean(np.abs((real_vals - pred_vals) / real_vals)) * 100
    confidence = 100 - mape
    accuracy = 100 - mape

    if combined_close > current_price * 1.01:
        recommended_action = "Buy"
    elif combined_close < current_price * 0.99:
        recommended_action = "Sell"
    else:
        recommended_action = "Hold"

    volatility_class = df["VolatilityClass"].iloc[-1] if "VolatilityClass" in df.columns else "N/A"

    table_df = pd.DataFrame({
        "Metric": [
            "Current Price",
            "CNN-LSTM Predicted Close",
            "Ensemble Prediction",
            "Model Weights Used",
            "Stop Loss",
            "Take Profit",
            "Confidence",
            "Accuracy",
            "Volatility",
            "Recommended Action"
        ],
        "Value": [
            f"{current_price:.2f}",
            f"{cnn_close:.2f}",
            f"{combined_close:.2f}",
            weight_ratio,
            f"{stop_loss:.2f}",
            f"{take_profit:.2f}",
            f"{confidence:.2f}%",
            f"{accuracy:.2f}%",
            volatility_class,
            recommended_action
        ]
    })

    st.subheader(f"📊 Model Predictions for {ticker}")
    st.table(table_df)

    # Optional: dummy forecast chart
    future_dates = pd.date_range(
        start=df.index[-1] + pd.Timedelta(minutes=15),
        periods=7*4,
        freq="15T"
    )
    future_prices = [combined_close + i * 0.1 for i in range(len(future_dates))]
    st.subheader("📈 1-Day Forecast Range")
    st.dataframe(pd.DataFrame({"Date": future_dates, "Predicted_Price": future_prices}))

###############################################################################
# 10) Main Streamlit
###############################################################################
st.title("Advanced AI Financial Predictions")

tickers = ["CC=F", "GC=F", "KC=F", "NG=F", "^GDAXI", "^HSI", "USD/JPY", "ETHUSD", "SOLUSD", "^SPX", "HG=F", "SI=F"]
ticker = st.sidebar.selectbox("Select a Ticker", tickers)

# (Optional) sentiment + google trends
fetch_news = st.sidebar.checkbox("Fetch News & Economic?", value=False)
fetch_trends = st.sidebar.checkbox("Fetch Google Trends?", value=False)

sent_score = 0.0
econ_value = 0.0
if fetch_news:
    try:
        # Adapts from your code if you have a real pipeline
        pass
    except:
        pass

trends_val = 0.0
if fetch_trends:
    try:
        # Adapt from your code
        pass
    except:
        pass

live_data = fetch_live_data([ticker])
if ticker not in live_data or live_data[ticker].empty:
    st.error(f"No data for {ticker}")
    st.stop()

df = live_data[ticker].copy()
df = enhance_features(df)
df["Sentiment"] = sent_score
df["Economic"]  = econ_value
df["GoogleTrends"] = trends_val
df = classify_volatility(df)

df.dropna(how='all', inplace=True)
if df.empty:
    st.error("After feature engineering, data is empty.")
    st.stop()

# Prepare single-time-step data for CNN+LSTM
X_1step, y_close, scaler_1step, feat_cols = prepare_single_time_data(df)
if X_1step is None:
    st.error("No numeric 'Close' in data. Can't predict.")
    st.stop()

df_preds = pd.DataFrame(index=df.index)

# Load & Predict CNN+LSTM
try:
    cnn_model = load_cnn_lstm_attention_model(ticker)
    pred_cnn = cnn_model.predict(X_1step).flatten()  # shape => (samples,)
    needed_cnn = len(pred_cnn)
    if needed_cnn > len(df):
        needed_cnn = len(df)
        pred_cnn = pred_cnn[:needed_cnn]

    df_preds_cnn = df.iloc[-needed_cnn:].copy()
    df_preds_cnn["CNN_LSTM_Predicted_Close"] = pred_cnn

    # Invert scaling
    scaled_arr2 = np.zeros((needed_cnn, len(feat_cols)))
    idx_for_pred_cnn = 0
    scaled_arr2[:, idx_for_pred_cnn] = pred_cnn
    inv_cnn = scaler_1step.inverse_transform(scaled_arr2)[:, idx_for_pred_cnn]
    df_preds_cnn["CNN_LSTM_Predicted_Close"] = inv_cnn

    df_preds = df_preds.merge(
        df_preds_cnn[["CNN_LSTM_Predicted_Close"]],
        left_index=True, right_index=True, how='left'
    )
except Exception as e:
    st.warning(f"CNN+LSTM not loaded: {e}")

# === Add other model predictions to df_preds before generating signals ===

try:
    gnn_series = predict_with_gnn(df)
    if not gnn_series.empty:
        df_preds["GNN_Predicted_Close"] = gnn_series
except Exception as e:
    st.warning(f"GNN error: {e}")

try:
    ode_ser = predict_with_neural_ode(df)
    if not ode_ser.empty:
        df_preds["NeuralODE_Close"] = ode_ser
except Exception as e:
    st.warning(f"Neural ODE error: {e}")

try:
    hmm_states = predict_with_hmm(df)
    if not hmm_states.empty:
        df_preds["HMM_Regime"] = hmm_states
except Exception as e:
    st.warning(f"HMM error: {e}")

try:
    action = predict_with_dqn(df, ticker)
    if action:
        df_preds.loc[df_preds.index[-1], "DQN_Action"] = action
except Exception as e:
    st.warning(f"DQN error: {e}")

# Display CNN predictions
display_cnn_predictions(df, df_preds, ticker)

# Trade Recommendation Logic Based on CNN
st.subheader("🧠 Final Trade Recommendation")
trade_metrics = generate_trade_decision(df, df_preds)
if "Error" in trade_metrics:
    st.warning(trade_metrics["Error"])
else:
    trade_table = pd.DataFrame({
        "Metric": list(trade_metrics.keys()),
        "Value": list(trade_metrics.values())
    })
    st.table(trade_table)

# GNN
st.subheader("GNN Predictions")
try:
    gnn_series = predict_with_gnn(df)
    if not gnn_series.empty:
        st.table(pd.DataFrame({
            "Metric": ["Last GNN Predicted Close"],
            "Value": [gnn_series.iloc[-1]]
        }))
    else:
        st.write("No GNN predictions available.")
except Exception as e:
    st.warning(f"GNN error: {e}")

# DQN
st.subheader("DQN Action Recommendation")
try:
    action = predict_with_dqn(df, ticker)
    if action:
        st.table(pd.DataFrame({"Metric":["DQN Suggests"], "Value":[action]}))
    else:
        st.write("No DQN action available.")
except Exception as e:
    st.warning(f"DQN error: {e}")

# HMM
st.subheader("HMM Market Regime")
try:
    hmm_states = predict_with_hmm(df)
    if not hmm_states.empty:
        st.write("Last 10 HMM states (table):")
        st.dataframe(hmm_states.tail(10))
        st.write(f"Most recent regime: {hmm_states.iloc[-1]}")
    else:
        st.write("No HMM states available.")
except Exception as e:
    st.warning(f"HMM error: {e}")

# Neural ODE
st.subheader("Neural ODE Forecast")
try:
    ode_ser = predict_with_neural_ode(df)
    if not ode_ser.empty:
        st.table(pd.DataFrame({
            "Metric":["Last Neural ODE Close"],
            "Value":[ode_ser.iloc[-1]]
        }))
    else:
        st.write("No Neural ODE forecast available.")
except Exception as e:
    st.warning(f"Neural ODE error: {e}")
