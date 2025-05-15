import os
os.makedirs("model", exist_ok=True)

import joblib
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Additional imports that may be required
# -------------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor  # as an example for AutoML
from sklearn.metrics import mean_squared_error

# Gym / Reinforcement Learning
import gym
from gym import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# For sentiment analysis and data fetching
from textblob import TextBlob  # (existing: unused if using transformers pipeline)
from pytrends.request import TrendReq
from transformers import pipeline  # Import pipeline from transformers

# For GNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data as GeoData
from torch_geometric.nn import GCNConv

# Optional for wavelet transform & chaos indicators
try:
    import pywt
except ImportError:
    pywt = None
    print("[WARN] pywt not installed. Fourier/Wavelet functions may be unavailable.")

# Optional for Hidden Markov Models
try:
    from hmmlearn import hmm
except ImportError:
    hmm = None
    print("[WARN] hmmlearn not installed. HMM functionality may be unavailable.")

# Optional for Neural ODE
try:
    from torchdiffeq import odeint
except ImportError:
    odeint = None
    print("[WARN] torchdiffeq not installed. Neural ODE functionality may be unavailable.")

# -------------------------------------------------------------------
# NEW: Keras / TensorFlow for CNN+LSTM+Attention & Transformer
# -------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Dense, Conv1D, LSTM, Dropout, Flatten, Layer,
    MultiHeadAttention, GlobalAveragePooling1D, Embedding
)
from tensorflow.keras.models import Sequential

# -------------------------------------------------------------------
# 0) Environment Variables & API Keys
# -------------------------------------------------------------------
def fetch_live_data(tickers, retries=3):
    """
    Fetch 15-minute historical data from FinancialModelingPrep.
    Requires FMP_API_KEY to be set in your environment.
    """
    data = {}
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set FMP_API_KEY in your environment.")

    for ticker in tickers:
        for attempt in range(retries):
            try:
                ticker_api = ticker.replace("/", "")
                url = f"https://financialmodelingprep.com/api/v3/historical-chart/15min/{ticker_api}?apikey={api_key}"
                response = requests.get(url)
                response.raise_for_status()
                data_json = response.json()

                if not data_json or len(data_json) < 1:
                    continue

                df = pd.DataFrame(data_json)
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                df.rename(
                    columns={
                        "close": "Close",
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "volume": "Volume",
                    },
                    inplace=True,
                )

                # Ensure 'Volume' is numeric
                df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

                # Sort by ascending datetime
                df.sort_index(inplace=True)

                data[ticker] = df
                break
            except Exception as e:
                if attempt < retries - 1:
                    continue
                else:
                    print(f"Error fetching data for {ticker}: {e}")
                    data[ticker] = pd.DataFrame()
        if ticker not in data:
            data[ticker] = pd.DataFrame()
    return data


def fetch_news_data():
    """
    Fetch news data from NewsAPI and economic data from FRED API.
    Requires NEWS_API_KEY_NEWSAPI and NEWS_API_KEY_FRED to be set in your environment.
    """
    news_api_key = os.getenv("NEWS_API_KEY_NEWSAPI")
    fred_api_key = os.getenv("NEWS_API_KEY_FRED")

    if not news_api_key:
        raise ValueError("NewsAPI key not found. Please set NEWS_API_KEY_NEWSAPI in your environment.")
    if not fred_api_key:
        raise ValueError("FRED API key not found. Please set NEWS_API_KEY_FRED in your environment.")

    # Fetch headlines from NewsAPI
    news_url = f'https://newsapi.org/v2/top-headlines?category=business&language=en&apiKey={news_api_key}'
    news_response = requests.get(news_url)
    if news_response.status_code != 200:
        raise ValueError("Failed to fetch news data from NewsAPI.")
    news_data = news_response.json()
    headlines = [article['title'] for article in news_data.get('articles', [])]

    # Fetch economic data from FRED API (e.g., interest rates)
    fred_series_id = 'FEDFUNDS'  # Effective Federal Funds Rate
    fred_url = f'https://api.stlouisfed.org/fred/series/observations?series_id={fred_series_id}&api_key={fred_api_key}&file_type=json'
    fred_response = requests.get(fred_url)
    if fred_response.status_code != 200:
        raise ValueError("Failed to fetch economic data from FRED API.")
    fred_data = fred_response.json()

    try:
        fred_observations = fred_data.get('observations', [])
        latest_fred_value = float(fred_observations[-1]['value']) if fred_observations else 0
    except (KeyError, IndexError, ValueError):
        latest_fred_value = 0

    return headlines, latest_fred_value


def compute_sentiment_score(headlines):
    """
    Compute average sentiment polarity of the news headlines using a transformer pipeline.
    """
    if not headlines:
        return 0.0

    sentiment_analyzer = pipeline('sentiment-analysis')
    sentiment_scores = []
    for headline in headlines:
        try:
            result = sentiment_analyzer(headline)[0]
            if result['label'] == 'POSITIVE':
                sentiment_scores.append(result['score'])
            elif result['label'] == 'NEGATIVE':
                sentiment_scores.append(-result['score'])
            else:
                sentiment_scores.append(0)
        except Exception as e:
            print(f"Error analyzing sentiment for headline: {headline}. Error: {e}")
            sentiment_scores.append(0)

    average_sentiment = np.mean(sentiment_scores)
    return average_sentiment


def fetch_google_trends_data(tickers):
    """
    Fetch Google Trends data for tickers using pytrends.
    If a 429 error occurs, sets the trend to 0.0 for that ticker.
    """
    pytrends = TrendReq()
    trends_data = {}
    for ticker in tickers:
        try:
            pytrends.build_payload([ticker], timeframe='now 7-d')
            interest_over_time_df = pytrends.interest_over_time()
            if not interest_over_time_df.empty:
                trend_score = interest_over_time_df[ticker].iloc[-1]
            else:
                trend_score = 0.0
        except Exception as e:
            print(f"Error fetching Google Trends data for {ticker}: {e}")
            trend_score = 0.0
        trends_data[ticker] = trend_score
    return trends_data


# -------------------------------------------------------------------
# Indicator Computations (Feature Engineering)
# -------------------------------------------------------------------
def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.replace([np.inf, -np.inf], np.nan)

def compute_MACD(series):
    exp1 = series.ewm(span=12, adjust=False).mean()
    exp2 = series.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=9, adjust=False).mean()
    return macd, signal_line

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
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = abs(df["High"] - df["Close"].shift(1))
    df["L-PC"] = abs(df["Low"] - df["Close"].shift(1))
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(window=period).mean()
    return df

def compute_ema(series, span=21):
    return series.ewm(span=span, adjust=False).mean()

def compute_OBV(df):
    df = df.copy()
    df["Direction"] = np.where(df["Close"] > df["Close"].shift(1), 1, -1)
    df["Direction"] = np.where(df["Close"] == df["Close"].shift(1), 0, df["Direction"])
    df["Volume"].fillna(0, inplace=True)
    df["OBV"] = (df["Volume"] * df["Direction"]).cumsum()
    return df

def compute_true_range(df):
    prev_close = df["Close"].shift(1)
    high_low = df["High"] - df["Low"]
    high_prev_close = np.abs(df["High"] - prev_close)
    low_prev_close = np.abs(df["Low"] - prev_close)
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    return tr

def compute_positive_DM(df):
    up_move = df["High"] - df["High"].shift(1)
    down_move = df["Low"].shift(1) - df["Low"]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    return plus_dm

def compute_negative_DM(df):
    up_move = df["High"] - df["High"].shift(1)
    down_move = df["Low"].shift(1) - df["Low"]
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    return minus_dm

def compute_ADX(df, n=14):
    df = df.copy()
    df["TR"] = compute_true_range(df)
    df["+DM"] = compute_positive_DM(df)
    df["-DM"] = compute_negative_DM(df)
    df["TRn"] = df["TR"].rolling(window=n).sum()
    df["+DMn"] = df["+DM"].rolling(window=n).sum()
    df["-DMn"] = df["-DM"].rolling(window=n).sum()
    df["+DI"] = 100 * (df["+DMn"] / df["TRn"])
    df["-DI"] = 100 * (df["-DMn"] / df["TRn"])
    df["DX"] = 100 * (np.abs(df["+DI"] - df["-DI"])) / (df["+DI"] + df["-DI"])
    df["ADX"] = df["DX"].rolling(window=n).mean()
    return df


# -------------------------------------------------------------------
# Enhanced Feature Engineering
# -------------------------------------------------------------------
def enhance_features(df):
    """
    Perform advanced feature engineering:
      - Lags
      - RSI, MACD, Stochastic, Bollinger, ATR, EMA(21), OBV, ADX
      - Short/Long MAs
      - Interaction feature
    """
    df = df.copy()
    # Lags
    df["Lag_1"] = df["Close"].shift(1)
    df["Lag_2"] = df["Close"].shift(2)
    df["Lag_3"] = df["Close"].shift(3)

    # RSI
    df["RSI"] = compute_RSI(df["Close"])

    # MACD
    df["MACD"], df["MACD_Signal"] = compute_MACD(df["Close"])

    # Stochastic
    df = compute_stochastic_oscillator(df)

    # Bollinger
    df = compute_bollinger_bands(df)

    # ATR
    df = compute_atr(df)

    # Moving Averages
    df["MA_Short"] = df["Close"].rolling(5).mean()
    df["MA_Long"] = df["Close"].rolling(20).mean()

    # Example interaction
    df["RSI_MACD_Interaction"] = df["RSI"] * df["MACD"]

    # EMA(21)
    df["EMA_21"] = compute_ema(df["Close"], span=21)

    # OBV
    df = compute_OBV(df)

    # ADX
    df = compute_ADX(df)

    # Drop NaNs
    df.dropna(inplace=True)
    return df


# -------------------------------------------------------------------
# DQN Reinforcement Learning
# -------------------------------------------------------------------
class TradingEnv(gym.Env):
    """
    Custom Environment that follows gym interface for trading.
    Actions: Buy(0), Sell(1), Hold(2)
    Observations: scaled features (including Sentiment/Economic columns)
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, scaler):
        super(TradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.scaler = scaler

        # Define action space: Buy(0), Sell(1), Hold(2)
        self.action_space = spaces.Discrete(3)

        # Because we've already inserted 'Sentiment' and 'Economic' into df,
        # the observation space corresponds directly to # of columns in df.
        num_features = self.df.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32
        )

        # Initial variables
        self.current_step = 0
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.num_shares = 0
        self.net_worth = self.initial_balance

        self.max_steps = len(self.df) - 1

    def _next_observation(self):
        try:
            obs = self.df.iloc[self.current_step].values.astype(np.float32)
        except Exception as e:
            print(
                f"[ERROR] Failed to get observation at step {self.current_step}: {e}")
            self.render()
            raise
        obs = self.scaler.transform([obs])[0]
        return obs.astype(np.float32)

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.max_steps

        price = self.df.loc[self.current_step, "Close"]
        reward = 0

        if action == 0:  # Buy
            max_shares = self.balance // price
            if max_shares > 0:
                self.num_shares += max_shares
                self.balance -= max_shares * price
        elif action == 1:  # Sell
            if self.num_shares > 0:
                self.balance += self.num_shares * price
                self.num_shares = 0

        # Update net worth
        self.net_worth = self.balance + self.num_shares * price
        reward = self.net_worth - self.initial_balance
        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.num_shares = 0

        return self._next_observation()

    def render(self, mode='human', close=False):
        profit = self.net_worth - self.initial_balance
        print(f"Step: {self.current_step}")
        print(f"Balance: {self.balance}")
        print(f"Shares held: {self.num_shares}")
        print(f"Net worth: {self.net_worth}")
        print(f"Profit: {profit}")
        print("DF columns:", self.df.columns)
        print("Sample row types at step 0:")
        print(self.df.iloc[0].apply(lambda x: type(x)))
        print("First few rows:")
        print(self.df.head())


def train_DQN(df, scaler, ticker):
    print(f"Training DQN agent for {ticker}...")

    env = TradingEnv(df, scaler)
    env = DummyVecEnv([lambda: env])
    dqn_model = DQN('MlpPolicy', env, verbose=1)

    try:
        dqn_model.learn(total_timesteps=10000)
    except Exception as e:
        print(f"[ERROR] Training failed for {ticker}: {e}")
        env.envs[0].render()
        raise

    model_filename = f"model/dqn_model_{ticker}"
    dqn_model.save(model_filename)
    print(f"DQN model saved to {model_filename}.zip")


# -------------------------------------------------------------------
# GNN – Implementation
# -------------------------------------------------------------------
def train_GNN(df):
    """
    Simple GNN implementation using PyTorch Geometric.
    Expects 'Close' to exist in df. Uses all other columns as features.
    """
    print("\n[ Training GNN Model ]")

    df = df.reset_index(drop=True)
    # We'll predict "Close" from the other columns
    features = df.drop(columns=['Close']).values
    targets = df['Close'].values

    # Create edges connecting consecutive time points
    edge_index = torch.tensor(
        [list(range(len(df) - 1)), list(range(1, len(df)))],
        dtype=torch.long
    )

    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(targets, dtype=torch.float)

    data = GeoData(x=x, edge_index=edge_index, y=y)

    class GNN(nn.Module):
        def __init__(self, num_features):
            super(GNN, self).__init__()
            self.conv1 = GCNConv(num_features, 64)
            self.conv2 = GCNConv(64, 32)
            self.fc1 = nn.Linear(32, 16)
            self.fc2 = nn.Linear(16, 1)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            return x.squeeze()

    model = GNN(num_features=x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "model/gnn_model.pth")
    print("GNN model saved to gnn_model.pth")


# -------------------------------------------------------------------
# Volatility Classification (High, Medium, Low)
# -------------------------------------------------------------------
def classify_volatility(df, lookback=14):
    """
    Classify volatility based on rolling standard deviation (or ATR).
    Return a new column 'VolatilityClass' with categories: 'Low', 'Medium', 'High'.
    """
    df = df.copy()
    if 'ATR' not in df.columns:
        df = compute_atr(df, period=lookback)

    # We'll do a rolling std of 'Close' as an alternative measure of volatility.
    df['RollingStd'] = df['Close'].rolling(lookback).std()

    # Combine ATR and RollingStd for a "VolatilityScore"
    df['VolatilityScore'] = 0.5 * df['ATR'] + 0.5 * df['RollingStd']

    # Classify into quantiles: [0-33%], (33%-66%], (66%-100%]
    vscore = df['VolatilityScore'].dropna()
    if not vscore.empty:
        low_thresh, high_thresh = vscore.quantile([0.33, 0.66])
    else:
        # fallback if not enough data
        low_thresh = high_thresh = 0

    conditions = [
        df['VolatilityScore'] <= low_thresh,
        (df['VolatilityScore'] > low_thresh) & (df['VolatilityScore'] <= high_thresh),
        df['VolatilityScore'] > high_thresh
    ]
    categories = ['Low', 'Medium', 'High']
    df['VolatilityClass'] = np.select(conditions, categories, default='Low')

    return df


# -------------------------------------------------------------------
# Autoencoders for Anomaly Detection
# -------------------------------------------------------------------
class Autoencoder(nn.Module):
    """
    Simple feedforward autoencoder for anomaly detection on price/feature data.
    """
    def __init__(self, input_dim, latent_dim=16):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed


def train_autoencoder(df):
    """
    Trains a simple autoencoder on the numeric columns of df for anomaly detection.
    Returns the trained model and the reconstruction errors.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number])
    x_data = numeric_cols.values
    x_tensor = torch.tensor(x_data, dtype=torch.float)

    # Build model
    input_dim = x_data.shape[1]
    model = Autoencoder(input_dim=input_dim, latent_dim=16)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    epochs = 20
    for epoch in range(epochs):
        optimizer.zero_grad()
        reconstructed = model(x_tensor)
        loss = criterion(reconstructed, x_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"[Autoencoder] Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Compute reconstruction errors
    model.eval()
    with torch.no_grad():
        reconstructed = model(x_tensor)
        mse = torch.mean((reconstructed - x_tensor) ** 2, dim=1).numpy()

    return model, mse


# -------------------------------------------------------------------
# Neural ODEs for Forecasting
# -------------------------------------------------------------------
class ODEFunc(nn.Module):
    """
    ODE function defining dX/dt = f(X(t), t)
    """
    def __init__(self, hidden_dim=16):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, t, x):
        return self.net(x)


def train_neural_ode(df, n_steps=50):
    """
    Train a simple Neural ODE for forecasting 'Close' prices.
    (Requires `torchdiffeq` to be installed.)
    """
    if odeint is None:
        print("[WARN] torchdiffeq not installed, skipping Neural ODE training.")
        return None

    close_prices = df['Close'].values.astype(np.float32)
    t = torch.linspace(0, 1, steps=len(close_prices))  # normalized time

    x = torch.tensor(close_prices).unsqueeze(-1)  # shape (len, 1)
    func = ODEFunc(hidden_dim=16)
    optimizer = torch.optim.Adam(func.parameters(), lr=0.001)

    print("[Neural ODE] Training... (this is a minimal/simplistic example)")

    for epoch in range(n_steps):
        optimizer.zero_grad()
        pred = odeint(func, x[0], t)
        pred = pred.squeeze()
        loss = F.mse_loss(pred, x.squeeze())
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

    return func


# -------------------------------------------------------------------
# Meta-Learning & Multi-Agent RL (Skeleton)
# -------------------------------------------------------------------
def meta_learning_skeleton():
    print("[Meta-Learning] This is a placeholder for advanced meta-learning training.")

def multi_agent_rl_skeleton():
    print("[Multi-Agent RL] This is a placeholder for multi-agent reinforcement learning.")


# -------------------------------------------------------------------
# Fourier/Wavelet Transforms, Chaos Theory Indicators
# -------------------------------------------------------------------
def compute_fft(series):
    if len(series) < 2:
        return None
    fft_result = np.fft.fft(series - np.mean(series))
    magnitude = np.abs(fft_result)
    return magnitude

def compute_wavelet_transform(series):
    if pywt is None:
        print("[WARN] pywt not installed, skipping wavelet transform.")
        return None, None
    wavelet = 'db1'
    level = 2
    coeffs = pywt.wavedec(series, wavelet, level=level)
    approx = coeffs[0]
    details = coeffs[1:]
    return approx, details

def chaos_theory_indicators_skeleton(series):
    print("[Chaos Theory] Placeholder for advanced chaos theory indicators.")


# -------------------------------------------------------------------
# 10) Hidden Markov Model for Market Regime Detection
# -------------------------------------------------------------------
def detect_market_regimes(df, n_states=2, covariance_type="full"):
    """
    Use a Hidden Markov Model (HMM) to detect market regimes based on 'Close' returns.
    - Replaces infinities with NaNs and fills them or drops them
    - Fits HMM to the returns
    - Adds a 'RegimeState' column to df
    """
    if hmm is None:
        print("[WARN] hmmlearn not installed, skipping HMM-based market regime detection.")
        return df, None

    df = df.copy()

    # 1) Compute returns
    df['Returns'] = df['Close'].pct_change()

    # 2) Clean up infinities/NaNs
    df['Returns'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['Returns'].fillna(0.0, inplace=True)

    # 3) Check variance
    std_returns = df['Returns'].std()
    if std_returns < 1e-12:
        print("[ERROR] Returns have near-zero variance. Cannot fit HMM.")
        return df, None

    # 4) Prepare data for HMM
    returns = df[['Returns']].values

    if len(returns) < n_states:
        print(f"[ERROR] Only {len(returns)} data points but attempting {n_states} states. Not enough data.")
        return df, None

    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=3000,   
        tol=1e-4,      # sometimes adjusting tolerance helps
        verbose=True   # see iteration-by-iteration log-likelihood
    )

    try:
        model.fit(returns)
    except ValueError as e:
        print(f"[ERROR] HMM fit failed: {e}")
        return df, None

    # 5) Predict regimes
    hidden_states = model.predict(returns)
    df['RegimeState'] = hidden_states

    return df, model


# -------------------------------------------------------------------
# 13) AutoML & Hyperparameter Optimization
# -------------------------------------------------------------------
def automl_optimization(df):
    print("[AutoML] Starting a simple hyperparameter optimization with GridSearchCV...")

    numeric_cols = df.select_dtypes(include=[np.number])
    if 'Close' not in numeric_cols.columns:
        print("No 'Close' column found in numeric data. Skipping.")
        return

    feature_cols = [c for c in numeric_cols.columns if c != 'Close']
    X = numeric_cols[feature_cols].values
    y = numeric_cols['Close'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    param_grid = {
        'hidden_layer_sizes': [(32,), (64,), (32, 32)],
        'alpha': [1e-5, 1e-4],
        'learning_rate_init': [0.001, 0.01]
    }
    mlp = MLPRegressor(max_iter=2000, random_state=42)
    grid_search = GridSearchCV(mlp, param_grid, scoring='neg_mean_squared_error', cv=3)
    grid_search.fit(X_train, y_train)

    print("[AutoML] Best params:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    predictions = best_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"[AutoML] MSE on test set: {mse:.4f}")


# -------------------------------------------------------------------
# --- NEW SECTION START ---
# 1) Custom Attention Layer for Keras (CNN+LSTM+Attention)
# -------------------------------------------------------------------
class AttentionLayer(Layer):
    """
    A simple, custom 'attention' layer that computes a weighted sum of LSTM outputs.
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight",
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias",
                                 shape=(input_shape[-1],),
                                 initializer="zeros")
        self.u = self.add_weight(name="att_u",
                                 shape=(input_shape[-1], 1),
                                 initializer="normal")
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x_tanh = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=[2, 0]) + self.b)
        score = tf.tensordot(x_tanh, self.u, axes=[2, 0])  # (batch_size, timesteps, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)  # sum across timesteps
        return context_vector


# -------------------------------------------------------------------
# 2) Keras model: CNN + LSTM + Attention
# -------------------------------------------------------------------
def train_cnn_lstm_attention(df, ticker, epochs=3):
    """
    Example function that constructs a CNN+LSTM+Attention model in Keras,
    then saves the model to {ticker}_cnn_lstm_attention_model_tuned.h5.
    """
    numeric_cols = df.select_dtypes(include=[np.number])
    if "Close" not in numeric_cols.columns:
        print(f"[{ticker} CNN-LSTM-Attn] No 'Close' in data, skipping.")
        return

    # For demonstration, treat each row as a single time step.
    # Real usage: reshape your data as (samples, timesteps, features).
    feature_cols = [c for c in numeric_cols.columns if c != "Close"]
    X = numeric_cols[feature_cols].values
    y = numeric_cols["Close"].values

    X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))  # (samples, 1, features)
    y_reshaped = y.reshape(-1, 1)

    # Build a small CNN+LSTM model with attention:
    inp = Input(shape=(X_reshaped.shape[1], X_reshaped.shape[2]))  # (1, features)

    # CNN
    x = Conv1D(filters=16, kernel_size=1, activation='relu')(inp)
    # LSTM
    x = LSTM(32, return_sequences=True)(x)
    # Attention
    x = AttentionLayer()(x)  # shape -> (batch_size, hidden_dim)
    # Dense
    x = Dense(16, activation='relu')(x)
    out = Dense(1, activation='linear')(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_reshaped, y_reshaped, epochs=epochs, batch_size=32, verbose=0)
    save_path = f"model/{ticker}_cnn_lstm_attention_model_tuned.h5"
    model.save(save_path)
    print(f"[{ticker}] CNN+LSTM+Attention model saved to {save_path}")



# -------------------------------------------------------------------
# 3) Keras Transformer model
# -------------------------------------------------------------------
def train_transformer_model(df, ticker, epochs=3):
    """
    Example function that constructs a simple Keras Transformer-like model,
    then saves the model to transformer_model_{ticker}.h5.
    """
    numeric_cols = df.select_dtypes(include=[np.number])
    if "Close" not in numeric_cols.columns:
        print(f"[{ticker} Transformer] No 'Close' in data, skipping.")
        return

    feature_cols = [c for c in numeric_cols.columns if c != "Close"]
    X = numeric_cols[feature_cols].values
    y = numeric_cols["Close"].values

    X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))  # (samples, 1, features)
    y_reshaped = y.reshape(-1, 1)

    # Minimal "Transformer" using Keras MultiHeadAttention:
    inp = Input(shape=(X_reshaped.shape[1], X_reshaped.shape[2]))  # (1, features)
    attn = MultiHeadAttention(num_heads=2, key_dim=X_reshaped.shape[2])(inp, inp)
    x = tf.reduce_mean(attn, axis=1)  # simple average across the single "time" dimension

    x = Dense(32, activation='relu')(x)
    out = Dense(1, activation='linear')(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_reshaped, y_reshaped, epochs=epochs, batch_size=32, verbose=0)
    save_name = f"transformer_model_{ticker}.h5"
    model.save(save_name)
    print(f"[{ticker}] Transformer model saved to {save_name}")


# -------------------------------------------------------------------
# 4) Utility to gather "metrics" + volatility ratio,
#    and to display "graph info" in table form
# -------------------------------------------------------------------
def compute_volatility_ratio(df):
    """
    Example function for 'volatility ratio' – ratio of std/mean of 'Close'.
    """
    if "Close" not in df.columns:
        return np.nan
    mean_close = df["Close"].mean()
    if mean_close == 0:
        return np.nan
    std_close = df["Close"].std()
    return std_close / mean_close

def create_metrics_table(ticker, df):
    """
    Returns a DataFrame with basic metrics, including the volatility ratio row,
    and also a placeholder for "graph info" as row data.
    """
    rows = []
    # Example placeholder for MSE from some model:
    mse_placeholder = np.random.rand()  # random number to illustrate
    volatility_ratio = compute_volatility_ratio(df)

    rows.append(("MSE (Placeholder)", mse_placeholder))
    rows.append(("Volatility Ratio", volatility_ratio))

    # "Graph info" in table form (e.g., min/max per column)
    for col in df.columns:
        col_min = df[col].min()
        col_max = df[col].max()
        rows.append((f"Graph Info for {col}", f"Min={col_min:.4f}, Max={col_max:.4f}"))

    metrics_df = pd.DataFrame(rows, columns=["Metric/Info", "Value"])
    return metrics_df


# -------------------------------------------------------------------
# 5) Weighted-Ratio Mechanism for Predictions
# -------------------------------------------------------------------
def compute_weighted_ensemble(df):
    """
    Example of a "weight ratio mechanism" to combine predictions
    from the CNN+Attention model and the Transformer model.
    
    This is just a demonstration. In practice, you'd handle:
      - Splitting data into training vs. test
      - Loading the saved models from .h5
      - Running model.predict(...) to get predictions
      - Combining them with your chosen weights
    """
    numeric_cols = df.select_dtypes(include=[np.number])
    if "Close" not in numeric_cols.columns:
        print("[Ensemble] 'Close' not found, skipping ensemble.")
        return None

    # We'll just do a trivial approach with the final row as "test" example:
    X = numeric_cols.drop(columns=["Close"]).values
    if len(X) < 1:
        print("[Ensemble] Not enough rows for predictions.")
        return None
    test_sample = X[-1:].reshape((1, 1, X.shape[1]))  # shape (1, timesteps=1, features)

    # Load the saved models (assuming they've been trained/saved for this ticker).
    # For demonstration, we pretend they're "transformer_model_XYZ.h5" or "XYZ_cnn_lstm_attention_model_tuned.h5"
    # If not found, we skip.
    # This is just a placeholder - adapt to your ticker name in production.
    try:
        cnn_lstm_model = tf.keras.models.load_model("demo_cnn_lstm_attention_model_tuned.h5",
                                                    custom_objects={"AttentionLayer": AttentionLayer})
        transformer_model = tf.keras.models.load_model("demo_transformer_model_XYZ.h5")
    except Exception as e:
        print(f"[Ensemble] Could not load one of the models: {e}")
        return None

    # Predict
    pred_cnn = cnn_lstm_model.predict(test_sample)[0][0]  # shape => single float
    pred_trans = transformer_model.predict(test_sample)[0][0]

    # Weighted average
    w1, w2 = 0.6, 0.4  # example weights
    final_prediction = w1 * pred_cnn + w2 * pred_trans

    print(f"[Ensemble] CNN+LSTM+Attn prediction = {pred_cnn:.4f}, Transformer prediction = {pred_trans:.4f}")
    print(f"[Ensemble] Weighted final prediction = {final_prediction:.4f}")
    return final_prediction


# -------------------------------------------------------------------
# 14) Main Execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    tickers = ["
        "CC=F", "GC=F", "KC=F", "NG=F", "^GDAXI", "^HSI", "USD/JPY", "ETHUSD", "SOLUSD", "^SPX", "HG=F", "SI=F", "CL=F"
    ]

    # Fetch News Data
    print("Fetching News and Economic Data...")
    try:
        headlines, economic_indicator = fetch_news_data()
        sentiment_score = compute_sentiment_score(headlines)
        print(f"Sentiment Score: {sentiment_score:.2f}")
        print(f"Economic Indicator (FEDFUNDS): {economic_indicator:.2f}%")
    except ValueError as e:
        print(f"Error fetching news or economic data: {e}")
        sentiment_score = 0
        economic_indicator = 0

    # Fetch Live Data
    print("Fetching Live Data...")
    data = fetch_live_data(tickers)

    # Fetch Google Trends Data
    print("Fetching Google Trends Data...")
    google_trends_data = fetch_google_trends_data(tickers)

    for ticker in tickers:
        if ticker not in data or data[ticker].empty:
            print(f"\nNo data for {ticker}, skipping.\n")
            continue

        df = data[ticker]
        print(f"\nFetched {len(df)} rows of data for {ticker}.")

        # Enhance features
        df = enhance_features(df)
        print(f"After enhance_features, {len(df)} rows remain (NaNs dropped).")

        if df.empty or len(df) < 10:
            print(f"[SKIP] {ticker} has too few rows after feature engineering.")
            continue

        # Drop non-numeric columns
        df = df.select_dtypes(include=[np.number])

        # Drop any rows with NaNs
        df.dropna(inplace=True)

        if df.empty or len(df) < 10:
            print(f"[SKIP] {ticker} data became empty after cleaning.")
            continue

        # Add Google Trends data
        df["GoogleTrends"] = google_trends_data.get(ticker, 0.0)
        # Add Sentiment/Economic
        df["Sentiment"] = sentiment_score
        df["Economic"] = economic_indicator

        # Classify volatility
        df = classify_volatility(df)

        # Keep numeric
        df = df.select_dtypes(include=[np.number])
        df.dropna(inplace=True)

        if df.empty or len(df) < 10:
            print(f"[SKIP] {ticker} data invalid after cleaning. Skipping.")
            continue

        print(f"[INFO] Cleaned df for {ticker}: {df.shape[0]} rows, {df.shape[1]} numeric columns")

        # Fit scaler
        scaler = MinMaxScaler()
        scaler.fit(df)

        # (A) Train a DQN Agent
        train_DQN(df, scaler, ticker)

        # (B) Train a GNN
        train_GNN(df)

        # (C) Autoencoder
        autoencoder_model, reconstruction_errors = train_autoencoder(df)
        if autoencoder_model is not None:
            torch.save(autoencoder_model.state_dict(), "model/autoencoder_model.pth")
            print("Autoencoder model saved to autoencoder_model.pth")

        # (D) Neural ODE
        ode_func = train_neural_ode(df)
        if ode_func is not None:
            torch.save(func.state_dict(), "model/neural_ode_func.pth")
            print("Neural ODE function saved to neural_ode_func.pth")

        # (E) Market Regime Detection (HMM)
        if hmm is not None:
            df_hmm, hmm_model = detect_market_regimes(df)
            if hmm_model is not None:
                joblib.dump(model, "model/hmm_model.pkl")
                print("HMM model saved to hmm_model.pkl")

        # (F) AutoML
        automl_optimization(df)

        # (G) Train Keras CNN+LSTM+Attention
        train_cnn_lstm_attention(df, ticker, epochs=3)

        # (H) Train Keras Transformer
        train_transformer_model(df, ticker, epochs=3)

        # Produce a "metrics table" with volatility ratio and "graph info"
        metrics_df = create_metrics_table(ticker, df)
        print(f"\n=== Metrics and Graph Info Table for {ticker} ===")
        print(metrics_df.to_string(index=False))

        # EXAMPLE: Weighted prediction demonstration (optional).
        # You'd adapt the paths and ticker in real usage.
        # compute_weighted_ensemble(df)

        print(f"\nAll methods completed for ticker: {ticker}\n")

    # Meta-Learning & Multi-Agent RL skeleton calls (not ticker-specific)
    meta_learning_skeleton()
    multi_agent_rl_skeleton()

    # Fourier/Wavelet example usage
    sample_ticker = tickers[0]
    if sample_ticker in data and not data[sample_ticker].empty:
        sample_series = data[sample_ticker]['Close']
        mag_spectrum = compute_fft(sample_series)
        approx, details = compute_wavelet_transform(sample_series)
        chaos_theory_indicators_skeleton(sample_series)

    print("\n[Completed all tasks in train.py]\n")
