"""
NQ Futures Price Prediction Neural Network
==========================================
Predicts NQ point movement over the next 15 bars (minutes) using a 90-bar lookback window.
Uses a CNN+LSTM hybrid architecture with overfitting protection and NVIDIA GPU auto-scaling.

Data: Databento .dbn file with 1-minute OHLCV bars (2021-2026)
Split: 80% train / 20% test (chronological)
"""

import os
import sys
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
LOOKBACK = 90  # bars to look back
HORIZON = 15  # bars to predict ahead
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10  # of training portion
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 15
LR_PATIENCE = 7
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
BASE_BATCH_SIZE = 64
DATA_FILE = "nq.dbn"
TRADE_THRESHOLD = 2.0  # minimum predicted point move to enter a trade
NQ_MULTIPLIER = 20.0  # $ per NQ point


# ─────────────────────────────────────────────
# GPU Setup & Auto-Scaling
# ─────────────────────────────────────────────
def setup_device():
    """Detect GPU, print info, and auto-scale batch size."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        gpu_mem_gb = props.total_mem / (1024**3)
        gpu_name = props.name
        print(f"GPU Detected: {gpu_name}")
        print(f"GPU Memory:   {gpu_mem_gb:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")

        # Auto-scale batch size based on GPU memory
        if gpu_mem_gb >= 24:
            batch_size = 256
        elif gpu_mem_gb >= 16:
            batch_size = 128
        elif gpu_mem_gb >= 8:
            batch_size = 64
        else:
            batch_size = 32

        print(f"Auto-scaled batch size: {batch_size}")
        use_amp = True
    else:
        device = torch.device("cpu")
        batch_size = BASE_BATCH_SIZE
        use_amp = False
        print("No GPU detected, using CPU")
        print(f"Batch size: {batch_size}")

    return device, batch_size, use_amp


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────
def load_data(filepath):
    """Load Databento .dbn file and return cleaned OHLCV DataFrame."""
    print(f"\nLoading data from {filepath}...")
    import databento as db

    store = db.DBNStore.from_path(filepath)
    df = store.to_df()

    print(f"Raw records: {len(df)}")

    # Databento stores prices as fixed-point integers (1e-9 scale)
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        if col in df.columns:
            # Check if prices look like fixed-point (very large numbers)
            if df[col].median() > 1e6:
                df[col] = df[col] / 1e9

    # Ensure volume column exists
    if "volume" not in df.columns and "size" in df.columns:
        df["volume"] = df["size"]

    # Keep only what we need
    keep_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep_cols].copy()

    # Drop rows with NaN or zero prices
    df = df.dropna()
    df = df[(df["close"] > 0) & (df["volume"] >= 0)]
    df = df.reset_index(drop=True)

    print(f"Clean records: {len(df)}")
    print(f"Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print(f"Date range: index 0 to {len(df)-1}")

    return df


# ─────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────
def build_features(df):
    """Create feature columns from raw OHLCV data."""
    feat = pd.DataFrame(index=df.index)

    # Price-based features
    feat["returns"] = df["close"].pct_change()
    feat["hl_range"] = df["high"] - df["low"]
    feat["bar_body"] = df["close"] - df["open"]
    feat["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    feat["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    # Volume features
    feat["volume"] = df["volume"]
    feat["vol_sma20"] = df["volume"].rolling(20).mean()
    feat["vol_ratio"] = df["volume"] / feat["vol_sma20"].replace(0, 1)

    # Moving average distances (relative to close)
    feat["sma10_dist"] = (df["close"] - df["close"].rolling(10).mean()) / df["close"]
    feat["sma30_dist"] = (df["close"] - df["close"].rolling(30).mean()) / df["close"]
    feat["sma60_dist"] = (df["close"] - df["close"].rolling(60).mean()) / df["close"]

    # RSI(14)
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    feat["rsi"] = 100 - (100 / (1 + rs))

    # Volatility
    feat["volatility"] = df["close"].pct_change().rolling(20).std()

    # OHLC normalized (as pct change from previous close)
    prev_close = df["close"].shift(1)
    feat["open_ret"] = (df["open"] - prev_close) / prev_close
    feat["high_ret"] = (df["high"] - prev_close) / prev_close
    feat["low_ret"] = (df["low"] - prev_close) / prev_close
    feat["close_ret"] = (df["close"] - prev_close) / prev_close

    # Drop NaN rows from rolling calculations
    feat = feat.dropna()

    return feat


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class NQDataset(Dataset):
    def __init__(self, features, targets):
        # features: (N, lookback, num_features) numpy array
        # targets: (N,) numpy array
        self.X = torch.FloatTensor(features)
        self.y = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_windows(features_df, close_prices, lookback, horizon):
    """Create sliding windows of features and corresponding targets."""
    feat_values = features_df.values
    close_values = close_prices.values
    n = len(feat_values)

    X_list = []
    y_list = []

    for i in range(lookback, n - horizon):
        X_list.append(feat_values[i - lookback : i])
        # Target: price change over next `horizon` bars
        y_list.append(close_values[i + horizon] - close_values[i])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    return X, y


# ─────────────────────────────────────────────
# Model: CNN + LSTM Hybrid
# ─────────────────────────────────────────────
class NQPredictor(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        # CNN layers to capture local patterns
        self.conv1 = nn.Conv1d(num_features, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.cnn_dropout = nn.Dropout(0.2)

        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )

        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x shape: (batch, seq_len=90, features)
        # Conv1d expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)

        x = self.cnn_dropout(torch.relu(self.bn1(self.conv1(x))))
        x = self.cnn_dropout(torch.relu(self.bn2(self.conv2(x))))

        # Back to (batch, seq_len, features) for LSTM
        x = x.permute(0, 2, 1)

        lstm_out, (h_n, _) = self.lstm(x)
        # Use last hidden state from the top LSTM layer
        last_hidden = h_n[-1]  # (batch, 128)

        out = self.fc(last_hidden)
        return out.squeeze(-1)


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
def train_model(model, train_loader, val_loader, device, use_amp):
    """Train with early stopping, LR scheduling, gradient clipping, and AMP."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=LR_PATIENCE, verbose=False
    )
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    print(f"\n{'Epoch':>5} | {'Train Loss':>12} | {'Val Loss':>12} | {'LR':>10} | {'Status'}")
    print("-" * 65)

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── Train ──
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                preds = model(X_batch)
                loss = criterion(preds, y_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())

        avg_train = np.mean(train_losses)

        # ── Validate ──
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    preds = model(X_batch)
                    loss = criterion(preds, y_batch)
                val_losses.append(loss.item())

        avg_val = np.mean(val_losses)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(avg_val)

        # ── Early stopping ──
        status = ""
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            status = "* best"
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"{epoch:>5} | {avg_train:>12.4f} | {avg_val:>12.4f} | {current_lr:>10.6f} | early stop")
                break

        if epoch % 5 == 0 or epoch == 1 or status:
            print(f"{epoch:>5} | {avg_train:>12.4f} | {avg_val:>12.4f} | {current_lr:>10.6f} | {status}")

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    return model


# ─────────────────────────────────────────────
# Backtest
# ─────────────────────────────────────────────
def backtest(model, test_loader, device, use_amp):
    """Run predictions on test set and compute metrics."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                preds = model(X_batch)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    # ── Metrics ──
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))

    # Directional accuracy
    pred_dir = np.sign(preds)
    actual_dir = np.sign(targets)
    dir_accuracy = np.mean(pred_dir == actual_dir) * 100

    # ── Simulated Trading ──
    equity = [0.0]
    trades = 0
    wins = 0

    for i in range(len(preds)):
        if abs(preds[i]) >= TRADE_THRESHOLD:
            trades += 1
            direction = np.sign(preds[i])
            pnl = direction * targets[i] * NQ_MULTIPLIER
            equity.append(equity[-1] + pnl)
            if pnl > 0:
                wins += 1
        else:
            equity.append(equity[-1])

    equity = np.array(equity)
    total_pnl = equity[-1]
    win_rate = (wins / trades * 100) if trades > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = peak - equity
    max_dd = np.max(drawdown)

    # ── Print Results ──
    print("\n" + "=" * 55)
    print("           BACKTEST RESULTS (Last 20%)")
    print("=" * 55)
    print(f"  Test samples:        {len(preds):,}")
    print(f"  MAE:                 {mae:.2f} points")
    print(f"  RMSE:                {rmse:.2f} points")
    print(f"  Directional Acc:     {dir_accuracy:.1f}%")
    print(f"  ─────────────────────────────────────")
    print(f"  Trades taken:        {trades:,} (threshold: {TRADE_THRESHOLD} pts)")
    print(f"  Win rate:            {win_rate:.1f}%")
    print(f"  Total P&L:           ${total_pnl:,.2f}")
    print(f"  Max Drawdown:        ${max_dd:,.2f}")
    print("=" * 55)

    # ── Plots ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Equity curve
    axes[0].plot(equity, linewidth=0.8)
    axes[0].set_title("Simulated Equity Curve")
    axes[0].set_xlabel("Trade #")
    axes[0].set_ylabel("Cumulative P&L ($)")
    axes[0].axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    axes[0].grid(True, alpha=0.3)

    # Predicted vs Actual
    sample_idx = np.random.choice(len(preds), min(2000, len(preds)), replace=False)
    axes[1].scatter(targets[sample_idx], preds[sample_idx], alpha=0.3, s=5)
    lims = [
        min(targets[sample_idx].min(), preds[sample_idx].min()),
        max(targets[sample_idx].max(), preds[sample_idx].max()),
    ]
    axes[1].plot(lims, lims, "r--", linewidth=1)
    axes[1].set_title("Predicted vs Actual (sample)")
    axes[1].set_xlabel("Actual Move (pts)")
    axes[1].set_ylabel("Predicted Move (pts)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("backtest_results.png", dpi=150)
    print(f"\nPlot saved to backtest_results.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  NQ Futures Prediction - CNN+LSTM Neural Network")
    print("=" * 55)

    # Setup
    device, batch_size, use_amp = setup_device()

    # Load data
    if not os.path.exists(DATA_FILE):
        print(f"\nERROR: {DATA_FILE} not found in current directory.")
        print("Place your Databento .dbn file here and try again.")
        sys.exit(1)

    df = load_data(DATA_FILE)

    # Build features
    print("\nEngineering features...")
    features_df = build_features(df)

    # Align close prices with features (features dropped some rows due to rolling)
    close_prices = df["close"].loc[features_df.index]

    # Create windowed samples
    print("Creating sliding windows...")
    X, y = create_windows(features_df, close_prices, LOOKBACK, HORIZON)
    print(f"Total samples: {len(y):,}")
    print(f"Feature shape: {X.shape} (samples, lookback, features)")

    # Chronological split
    split_idx = int(len(y) * TRAIN_RATIO)
    X_train_full, X_test = X[:split_idx], X[split_idx:]
    y_train_full, y_test = y[:split_idx], y[split_idx:]

    # Validation split from training data
    val_split = int(len(X_train_full) * (1 - VAL_RATIO))
    X_train, X_val = X_train_full[:val_split], X_train_full[val_split:]
    y_train, y_val = y_train_full[:val_split], y_train_full[val_split:]

    print(f"Train: {len(y_train):,} | Val: {len(y_val):,} | Test: {len(y_test):,}")

    # Normalize features using training stats only
    num_features = X_train.shape[2]
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, num_features)
    scaler.fit(X_train_flat)

    X_train = scaler.transform(X_train.reshape(-1, num_features)).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, num_features)).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, num_features)).reshape(X_test.shape)

    # Replace any NaN/inf from normalization
    for arr in [X_train, X_val, X_test]:
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # DataLoaders
    pin = device.type == "cuda"
    num_workers = 4 if device.type == "cuda" else 0
    train_loader = DataLoader(
        NQDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        NQDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        NQDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    # Model
    model = NQPredictor(num_features=num_features).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    # Train
    print("\nStarting training...")
    start_time = time.time()
    model = train_model(model, train_loader, val_loader, device, use_amp)
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed / 60:.1f} minutes")

    # Backtest
    backtest(model, test_loader, device, use_amp)

    # Save model
    torch.save(model.state_dict(), "nq_model.pt")
    print("Model saved to nq_model.pt")


if __name__ == "__main__":
    main()
