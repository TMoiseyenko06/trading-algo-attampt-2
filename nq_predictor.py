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
import argparse
import multiprocessing
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import entropy as scipy_entropy

# Prevent fork() deadlock warnings with CUDA + multiprocessing
multiprocessing.set_start_method("spawn", force=True)
from sklearn.preprocessing import StandardScaler
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def box(title, lines, width=62):
    """Print a formatted box with title and key-value lines."""
    print()
    print("┌" + "─" * width + "┐")
    print("│" + title.center(width) + "│")
    print("├" + "─" * width + "┤")
    for line in lines:
        if line == "---":
            print("├" + "─" * width + "┤")
        else:
            print("│  " + line.ljust(width - 2) + "│")
    print("└" + "─" * width + "┘")

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
LOOKBACK = 90  # bars to look back
HORIZON = 15  # bars to predict ahead
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10  # of training portion
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 25
LR_PATIENCE = 7
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 5e-4
GRAD_CLIP = 1.0
BASE_BATCH_SIZE = 64
DATA_FILE = "nq.dbn"
CHECKPOINT_FILE = "nq_checkpoint.pt"
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
        gpu_mem_gb = props.total_memory / (1024**3)
        gpu_name = props.name

        # Auto-scale batch size based on GPU memory
        if gpu_mem_gb >= 24:
            batch_size = 256
        elif gpu_mem_gb >= 16:
            batch_size = 128
        elif gpu_mem_gb >= 8:
            batch_size = 64
        else:
            batch_size = 32

        use_amp = True
        box("SYSTEM", [
            f"Device:          CUDA ({gpu_name})",
            f"GPU Memory:      {gpu_mem_gb:.1f} GB",
            f"CUDA Version:    {torch.version.cuda}",
            f"PyTorch:         {torch.__version__}",
            f"Batch Size:      {batch_size}  (auto-scaled)",
            f"Mixed Precision: Enabled (AMP)",
        ])
    else:
        device = torch.device("cpu")
        batch_size = BASE_BATCH_SIZE
        use_amp = False
        box("SYSTEM", [
            f"Device:          CPU",
            f"PyTorch:         {torch.__version__}",
            f"Batch Size:      {batch_size}",
            f"Mixed Precision: Disabled",
        ])

    return device, batch_size, use_amp


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────
def load_data(filepath):
    """Load Databento .dbn file and return cleaned OHLCV DataFrame."""
    print(f"\nLoading data from {filepath}...")
    import databento as db

    store = db.DBNStore.from_file(filepath)
    df = store.to_df()

    raw_count = len(store.to_df())

    df = store.to_df()

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

    # Data statistics
    avg_range = (df["high"] - df["low"]).mean()
    avg_volume = df["volume"].mean()
    total_days = len(df) / 390  # approx trading minutes per day

    box("DATA LOADED", [
        f"Source:          {filepath}",
        f"Raw Records:     {raw_count:,}",
        f"Clean Records:   {len(df):,}  ({len(df) - raw_count:+,} filtered)",
        f"Approx Days:     {total_days:,.0f}",
        "---",
        f"Price Range:     {df['close'].min():.2f}  →  {df['close'].max():.2f}",
        f"Avg Bar Range:   {avg_range:.2f} pts",
        f"Avg Volume:      {avg_volume:,.0f}",
        f"Open:            {df['open'].iloc[0]:.2f}  →  {df['open'].iloc[-1]:.2f}",
        f"Close:           {df['close'].iloc[0]:.2f}  →  {df['close'].iloc[-1]:.2f}",
    ])

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
    """Create sliding windows of features and corresponding targets.
    Also returns bar-by-bar price paths for SL/TP simulation."""
    feat_values = features_df.values
    close_values = close_prices.values
    n = len(feat_values)

    X_list = []
    y_list = []
    paths_list = []

    for i in range(lookback, n - horizon):
        X_list.append(feat_values[i - lookback : i])
        # Target: price change over next `horizon` bars
        y_list.append(close_values[i + horizon] - close_values[i])
        # Bar-by-bar path: price change at each bar relative to entry
        paths_list.append(close_values[i + 1 : i + horizon + 1] - close_values[i])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    paths = np.array(paths_list, dtype=np.float32)  # (n_samples, horizon)

    return X, y, paths


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
        self.cnn_dropout = nn.Dropout(0.4)

        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.4,
        )

        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
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
        optimizer, mode="min", factor=0.5, patience=LR_PATIENCE
    )
    criterion = nn.HuberLoss(delta=1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0
    history = {"train": [], "val": []}

    print()
    print("┌───────┬──────────────┬──────────────┬────────────┬───────────┐")
    print("│ Epoch │  Train Loss  │   Val Loss   │     LR     │  Status   │")
    print("├───────┼──────────────┼──────────────┼────────────┼───────────┤")

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

        history["train"].append(avg_train)
        history["val"].append(avg_val)

        # ── Early stopping ──
        status = ""
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            status = "★ best"
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"│  {epoch:>3}  │  {avg_train:>10.4f}  │  {avg_val:>10.4f}  │  {current_lr:>8.6f}  │  STOP     │")
                break

        if epoch % 5 == 0 or epoch == 1 or status:
            print(f"│  {epoch:>3}  │  {avg_train:>10.4f}  │  {avg_val:>10.4f}  │  {current_lr:>8.6f}  │  {status:<7}  │")

    print("└───────┴──────────────┴──────────────┴────────────┴───────────┘")

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # Overfit gap
    if len(history["train"]) > 0:
        final_train = history["train"][-1]
        overfit_ratio = best_val_loss / final_train if final_train > 0 else 0
        box("TRAINING SUMMARY", [
            f"Best Epoch:      {best_epoch} / {epoch}",
            f"Best Val Loss:   {best_val_loss:.4f}",
            f"Final Train Loss:{final_train:.4f}",
            f"Overfit Ratio:   {overfit_ratio:.3f}  (val/train, ~1.0 = good)",
            f"Early Stopped:   {'Yes' if patience_counter >= EARLY_STOP_PATIENCE else 'No'}",
        ])

    return model, history


# ─────────────────────────────────────────────
# Backtest
# ─────────────────────────────────────────────
def compute_prediction_entropy(preds, n_bins=50):
    """Compute Shannon entropy of prediction distribution (higher = more spread)."""
    hist, _ = np.histogram(preds, bins=n_bins, density=True)
    hist = hist[hist > 0]  # remove zeros for log
    hist = hist / hist.sum()  # normalize to probability
    return scipy_entropy(hist, base=2)


def backtest_sltp(model, test_loader, device, use_amp, y_mean, y_std, test_paths, tp_pct, sl_pct):
    """Backtest with stop-loss and take-profit levels based on prediction magnitude.

    TP = abs(prediction) * tp_pct
    SL = TP * sl_pct

    Walks bar-by-bar through each 15-bar window to check if TP or SL is hit first.
    If neither is hit, the trade exits at the final bar (hold to expiry).
    """
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

    # Denormalize predictions back to real NQ points
    preds = preds * y_std + y_mean

    # ── Simulate SL/TP trades ──
    equity = [0.0]
    trade_pnls = []
    long_pnls = []
    short_pnls = []
    tp_hits = 0
    sl_hits = 0
    expiry_exits = 0

    for i in range(len(preds)):
        if abs(preds[i]) < TRADE_THRESHOLD:
            equity.append(equity[-1])
            continue

        direction = np.sign(preds[i])
        tp_level = abs(preds[i]) * tp_pct   # TP in points
        sl_level = tp_level * sl_pct         # SL in points

        # Walk bar-by-bar through the price path
        path = test_paths[i]  # (horizon,) — price change at each bar vs entry
        exit_pnl = None

        for bar in range(len(path)):
            move_in_direction = direction * path[bar]  # positive = favorable

            if move_in_direction >= tp_level:
                # TP hit — exit at TP level
                exit_pnl = tp_level * NQ_MULTIPLIER
                tp_hits += 1
                break
            elif move_in_direction <= -sl_level:
                # SL hit — exit at SL level
                exit_pnl = -sl_level * NQ_MULTIPLIER
                sl_hits += 1
                break

        if exit_pnl is None:
            # Neither hit — exit at final bar (hold to expiry)
            exit_pnl = direction * path[-1] * NQ_MULTIPLIER
            expiry_exits += 1

        equity.append(equity[-1] + exit_pnl)
        trade_pnls.append(exit_pnl)
        if direction > 0:
            long_pnls.append(exit_pnl)
        else:
            short_pnls.append(exit_pnl)

    equity = np.array(equity)
    trade_pnls = np.array(trade_pnls) if trade_pnls else np.array([0.0])
    long_pnls = np.array(long_pnls) if long_pnls else np.array([0.0])
    short_pnls = np.array(short_pnls) if short_pnls else np.array([0.0])

    total_trades = len(trade_pnls)
    wins = np.sum(trade_pnls > 0)
    losses = np.sum(trade_pnls < 0)
    breakeven = np.sum(trade_pnls == 0)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    total_pnl = equity[-1]

    avg_win = np.mean(trade_pnls[trade_pnls > 0]) if wins > 0 else 0
    avg_loss = np.mean(trade_pnls[trade_pnls < 0]) if losses > 0 else 0
    largest_win = np.max(trade_pnls) if total_trades > 0 else 0
    largest_loss = np.min(trade_pnls) if total_trades > 0 else 0
    profit_factor = abs(np.sum(trade_pnls[trade_pnls > 0]) / np.sum(trade_pnls[trade_pnls < 0])) if losses > 0 and np.sum(trade_pnls[trade_pnls < 0]) != 0 else float("inf")
    expectancy = np.mean(trade_pnls) if total_trades > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = peak - equity
    max_dd = np.max(drawdown)
    max_dd_pct = (max_dd / np.max(peak) * 100) if np.max(peak) > 0 else 0

    # Sharpe
    if total_trades > 1 and np.std(trade_pnls) > 0:
        sharpe = np.mean(trade_pnls) / np.std(trade_pnls) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Long/short breakdown
    long_total = len(long_pnls)
    long_wins = np.sum(long_pnls > 0)
    long_wr = (long_wins / long_total * 100) if long_total > 0 else 0
    long_pnl_total = np.sum(long_pnls)

    short_total = len(short_pnls)
    short_wins = np.sum(short_pnls > 0)
    short_wr = (short_wins / short_total * 100) if short_total > 0 else 0
    short_pnl_total = np.sum(short_pnls)

    # ── Print Results ──
    box("SL/TP BACKTEST", [
        f"TP:  {tp_pct*100:.0f}% of prediction  |  SL:  {sl_pct*100:.0f}% of TP",
        f"Trade Threshold:           {TRADE_THRESHOLD} pts  |  NQ Multiplier: ${NQ_MULTIPLIER:.0f}/pt",
        "---",
        f"Total Trades:              {total_trades:,}",
        f"  Wins:                    {int(wins):,}",
        f"  Losses:                  {int(losses):,}",
        f"  Breakeven:               {int(breakeven):,}",
        f"Win Rate:                  {win_rate:.1f}%",
        "---",
        f"Exit Reasons:",
        f"  TP Hit:                  {tp_hits:,}  ({tp_hits/total_trades*100:.1f}%)" if total_trades > 0 else f"  TP Hit:                  0",
        f"  SL Hit:                  {sl_hits:,}  ({sl_hits/total_trades*100:.1f}%)" if total_trades > 0 else f"  SL Hit:                  0",
        f"  Held to Expiry:          {expiry_exits:,}  ({expiry_exits/total_trades*100:.1f}%)" if total_trades > 0 else f"  Held to Expiry:          0",
        "---",
        f"Total P&L:                 ${total_pnl:>12,.2f}",
        f"Avg Win:                   ${avg_win:>12,.2f}",
        f"Avg Loss:                  ${avg_loss:>12,.2f}",
        f"Largest Win:               ${largest_win:>12,.2f}",
        f"Largest Loss:              ${largest_loss:>12,.2f}",
        f"Expectancy (per trade):    ${expectancy:>12,.2f}",
        f"Profit Factor:             {profit_factor:>12.2f}",
        "---",
        f"Max Drawdown:              ${max_dd:>12,.2f}  ({max_dd_pct:.1f}%)",
        f"Annualized Sharpe:         {sharpe:>12.3f}",
    ])

    box("LONG vs SHORT BREAKDOWN (SL/TP)", [
        f"{'':28} {'Long':>15} {'Short':>15}",
        f"{'Trades:':28} {long_total:>15,} {short_total:>15,}",
        f"{'Win Rate:':28} {long_wr:>14.1f}% {short_wr:>14.1f}%",
        f"{'Total P&L:':28} ${long_pnl_total:>13,.2f} ${short_pnl_total:>13,.2f}",
    ])

    # ── Equity Curve Plot ──
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(equity, linewidth=0.8, color="#2196F3")
    ax.fill_between(range(len(equity)), equity, 0, alpha=0.1, color="#2196F3")
    ax.set_title(f"SL/TP Equity Curve  (TP={tp_pct*100:.0f}%, SL={sl_pct*100:.0f}% of TP)")
    ax.set_xlabel("Bar")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("backtest_sltp.png", dpi=150)
    print(f"\nPlot saved to backtest_sltp.png")


def backtest(model, test_loader, device, use_amp, y_mean=0.0, y_std=1.0, history=None):
    """Run predictions on test set and compute comprehensive metrics."""
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

    # Denormalize predictions back to real NQ points
    # (targets are already in real points since y_test was not normalized)
    preds = preds * y_std + y_mean

    # ── Prediction Distribution Stats ──
    pred_mean = np.mean(preds)
    pred_std = np.std(preds)
    pred_min = np.min(preds)
    pred_max = np.max(preds)
    pred_median = np.median(preds)
    pred_entropy = compute_prediction_entropy(preds)

    # ── Target Distribution Stats ──
    tgt_mean = np.mean(targets)
    tgt_std = np.std(targets)
    tgt_min = np.min(targets)
    tgt_max = np.max(targets)
    tgt_median = np.median(targets)
    tgt_entropy = compute_prediction_entropy(targets)

    # ── Error Metrics ──
    errors = preds - targets
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    mape = np.mean(np.abs(errors) / (np.abs(targets) + 1e-8)) * 100
    correlation = np.corrcoef(preds, targets)[0, 1]
    r_squared = 1 - np.sum(errors**2) / np.sum((targets - tgt_mean) ** 2)

    # ── Directional Accuracy ──
    pred_dir = np.sign(preds)
    actual_dir = np.sign(targets)
    dir_accuracy = np.mean(pred_dir == actual_dir) * 100
    # Breakdown by direction
    long_mask = pred_dir > 0
    short_mask = pred_dir < 0
    long_acc = np.mean(actual_dir[long_mask] > 0) * 100 if long_mask.sum() > 0 else 0
    short_acc = np.mean(actual_dir[short_mask] < 0) * 100 if short_mask.sum() > 0 else 0
    long_count = long_mask.sum()
    short_count = short_mask.sum()
    flat_count = (pred_dir == 0).sum()

    # ── Simulated Trading ──
    equity = [0.0]
    trade_pnls = []
    long_pnls = []
    short_pnls = []

    for i in range(len(preds)):
        if abs(preds[i]) >= TRADE_THRESHOLD:
            direction = np.sign(preds[i])
            pnl = direction * targets[i] * NQ_MULTIPLIER
            equity.append(equity[-1] + pnl)
            trade_pnls.append(pnl)
            if direction > 0:
                long_pnls.append(pnl)
            else:
                short_pnls.append(pnl)
        else:
            equity.append(equity[-1])

    equity = np.array(equity)
    trade_pnls = np.array(trade_pnls) if trade_pnls else np.array([0.0])
    long_pnls = np.array(long_pnls) if long_pnls else np.array([0.0])
    short_pnls = np.array(short_pnls) if short_pnls else np.array([0.0])

    total_trades = len(trade_pnls)
    wins = np.sum(trade_pnls > 0)
    losses = np.sum(trade_pnls < 0)
    breakeven = np.sum(trade_pnls == 0)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    total_pnl = equity[-1]

    avg_win = np.mean(trade_pnls[trade_pnls > 0]) if wins > 0 else 0
    avg_loss = np.mean(trade_pnls[trade_pnls < 0]) if losses > 0 else 0
    largest_win = np.max(trade_pnls) if total_trades > 0 else 0
    largest_loss = np.min(trade_pnls) if total_trades > 0 else 0
    profit_factor = abs(np.sum(trade_pnls[trade_pnls > 0]) / np.sum(trade_pnls[trade_pnls < 0])) if losses > 0 and np.sum(trade_pnls[trade_pnls < 0]) != 0 else float("inf")
    expectancy = np.mean(trade_pnls) if total_trades > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = peak - equity
    max_dd = np.max(drawdown)
    max_dd_pct = (max_dd / np.max(peak) * 100) if np.max(peak) > 0 else 0

    # Sharpe-like ratio (on trade returns)
    if total_trades > 1 and np.std(trade_pnls) > 0:
        sharpe = np.mean(trade_pnls) / np.std(trade_pnls) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Long/short breakdown
    long_wins = np.sum(long_pnls > 0)
    long_total = len(long_pnls)
    long_wr = (long_wins / long_total * 100) if long_total > 0 else 0
    long_pnl_total = np.sum(long_pnls)

    short_wins = np.sum(short_pnls > 0)
    short_total = len(short_pnls)
    short_wr = (short_wins / short_total * 100) if short_total > 0 else 0
    short_pnl_total = np.sum(short_pnls)

    # ── Print Results ──
    box("PREDICTION DISTRIBUTION", [
        f"{'':30} {'Predictions':>14} {'Actual':>14}",
        f"{'Mean:':30} {pred_mean:>14.4f} {tgt_mean:>14.4f}",
        f"{'Std Dev:':30} {pred_std:>14.4f} {tgt_std:>14.4f}",
        f"{'Median:':30} {pred_median:>14.4f} {tgt_median:>14.4f}",
        f"{'Min:':30} {pred_min:>14.4f} {tgt_min:>14.4f}",
        f"{'Max:':30} {pred_max:>14.4f} {tgt_max:>14.4f}",
        f"{'Entropy (bits):':30} {pred_entropy:>14.3f} {tgt_entropy:>14.3f}",
    ])

    # Magnitude accuracy: % within various point thresholds
    abs_errors = np.abs(errors)
    within_5 = np.mean(abs_errors <= 5) * 100
    within_10 = np.mean(abs_errors <= 10) * 100
    within_25 = np.mean(abs_errors <= 25) * 100
    within_50 = np.mean(abs_errors <= 50) * 100

    # Combined: correct direction AND within threshold
    correct_dir_mask = pred_dir == actual_dir
    dir_and_5 = np.mean(correct_dir_mask & (abs_errors <= 5)) * 100
    dir_and_10 = np.mean(correct_dir_mask & (abs_errors <= 10)) * 100
    dir_and_25 = np.mean(correct_dir_mask & (abs_errors <= 25)) * 100
    dir_and_50 = np.mean(correct_dir_mask & (abs_errors <= 50)) * 100

    box("MODEL ACCURACY", [
        f"Test Samples:              {len(preds):,}",
        f"MAE:                       {mae:.4f} pts",
        f"RMSE:                      {rmse:.4f} pts",
        f"MAPE:                      {mape:.2f}%",
        f"Correlation (r):           {correlation:.4f}",
        f"R-squared:                 {r_squared:.4f}",
        "---",
        f"Directional Accuracy:      {dir_accuracy:.1f}%",
        f"  Long predictions:        {long_count:,}  (acc: {long_acc:.1f}%)",
        f"  Short predictions:       {short_count:,}  (acc: {short_acc:.1f}%)",
        f"  Flat predictions:        {flat_count:,}",
        "---",
        f"{'Threshold':20} {'Within':>12} {'Dir + Within':>14}",
        f"{'  ± 5  pts':20} {within_5:>11.1f}% {dir_and_5:>13.1f}%",
        f"{'  ± 10 pts':20} {within_10:>11.1f}% {dir_and_10:>13.1f}%",
        f"{'  ± 25 pts':20} {within_25:>11.1f}% {dir_and_25:>13.1f}%",
        f"{'  ± 50 pts':20} {within_50:>11.1f}% {dir_and_50:>13.1f}%",
    ])

    box("TRADING PERFORMANCE", [
        f"Trade Threshold:           {TRADE_THRESHOLD} pts  |  NQ Multiplier: ${NQ_MULTIPLIER:.0f}/pt",
        "---",
        f"Total Trades:              {total_trades:,}",
        f"  Wins:                    {int(wins):,}",
        f"  Losses:                  {int(losses):,}",
        f"  Breakeven:               {int(breakeven):,}",
        f"Win Rate:                  {win_rate:.1f}%",
        "---",
        f"Total P&L:                 ${total_pnl:>12,.2f}",
        f"Avg Win:                   ${avg_win:>12,.2f}",
        f"Avg Loss:                  ${avg_loss:>12,.2f}",
        f"Largest Win:               ${largest_win:>12,.2f}",
        f"Largest Loss:              ${largest_loss:>12,.2f}",
        f"Expectancy (per trade):    ${expectancy:>12,.2f}",
        f"Profit Factor:             {profit_factor:>12.2f}",
        "---",
        f"Max Drawdown:              ${max_dd:>12,.2f}  ({max_dd_pct:.1f}%)",
        f"Annualized Sharpe:         {sharpe:>12.3f}",
    ])

    box("LONG vs SHORT BREAKDOWN", [
        f"{'':28} {'Long':>15} {'Short':>15}",
        f"{'Trades:':28} {long_total:>15,} {short_total:>15,}",
        f"{'Win Rate:':28} {long_wr:>14.1f}% {short_wr:>14.1f}%",
        f"{'Total P&L:':28} ${long_pnl_total:>13,.2f} ${short_pnl_total:>13,.2f}",
    ])

    # ── Plots ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("NQ Prediction Backtest Results", fontsize=14, fontweight="bold")

    # 1. Equity curve
    axes[0, 0].plot(equity, linewidth=0.8, color="#2196F3")
    axes[0, 0].fill_between(range(len(equity)), equity, 0, alpha=0.1, color="#2196F3")
    axes[0, 0].set_title("Equity Curve")
    axes[0, 0].set_xlabel("Bar")
    axes[0, 0].set_ylabel("Cumulative P&L ($)")
    axes[0, 0].axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Predicted vs Actual scatter
    sample_n = min(3000, len(preds))
    sample_idx = np.random.choice(len(preds), sample_n, replace=False)
    axes[0, 1].scatter(targets[sample_idx], preds[sample_idx], alpha=0.2, s=4, c="#FF5722")
    lims = [
        min(targets[sample_idx].min(), preds[sample_idx].min()),
        max(targets[sample_idx].max(), preds[sample_idx].max()),
    ]
    axes[0, 1].plot(lims, lims, "k--", linewidth=1, alpha=0.5)
    axes[0, 1].set_title(f"Predicted vs Actual  (r={correlation:.3f})")
    axes[0, 1].set_xlabel("Actual Move (pts)")
    axes[0, 1].set_ylabel("Predicted Move (pts)")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Prediction distribution histogram
    axes[1, 0].hist(targets, bins=80, alpha=0.5, label="Actual", color="#4CAF50", density=True)
    axes[1, 0].hist(preds, bins=80, alpha=0.5, label="Predicted", color="#FF9800", density=True)
    axes[1, 0].set_title("Distribution: Predicted vs Actual")
    axes[1, 0].set_xlabel("Point Move")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Training loss curve (if available)
    if history and "train" in history and len(history["train"]) > 0:
        epochs_range = range(1, len(history["train"]) + 1)
        axes[1, 1].plot(epochs_range, history["train"], label="Train", linewidth=1.2, color="#2196F3")
        axes[1, 1].plot(epochs_range, history["val"], label="Val", linewidth=1.2, color="#F44336")
        axes[1, 1].set_title("Training & Validation Loss")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("MSE Loss")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # Trade P&L distribution
        if total_trades > 1:
            axes[1, 1].hist(trade_pnls, bins=60, color="#9C27B0", alpha=0.7)
            axes[1, 1].axvline(x=0, color="red", linestyle="--", linewidth=1)
            axes[1, 1].set_title("Trade P&L Distribution")
            axes[1, 1].set_xlabel("P&L ($)")
            axes[1, 1].set_ylabel("Count")
            axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("backtest_results.png", dpi=150)
    print(f"\nPlot saved to backtest_results.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="NQ Futures Price Prediction Neural Network")
    parser.add_argument(
        "--resume", type=str, default=None, metavar="PATH",
        help=f"Resume training from a saved checkpoint (default: None). "
             f"Use --resume {CHECKPOINT_FILE} to load the last saved checkpoint.",
    )
    parser.add_argument(
        "--backtest-only", action="store_true",
        help="Skip training and only run backtest using saved checkpoint.",
    )
    parser.add_argument(
        "--tp", type=float, default=None, metavar="PCT",
        help="Take-profit as a fraction of the prediction (e.g. 0.5 = 50%%). "
             "Requires --sl. Runs SL/TP backtest mode.",
    )
    parser.add_argument(
        "--sl", type=float, default=None, metavar="PCT",
        help="Stop-loss as a fraction of the TP level (e.g. 0.5 = 50%% of TP). "
             "Requires --tp.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║        NQ FUTURES PREDICTION — CNN+LSTM NEURAL NETWORK     ║")
    print("║        Lookback: 90 bars  |  Horizon: 15 bars              ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Setup
    device, batch_size, use_amp = setup_device()

    # Load data
    if not os.path.exists(DATA_FILE):
        print(f"\nERROR: {DATA_FILE} not found in current directory.")
        print("Place your Databento .dbn file here and try again.")
        sys.exit(1)

    df = load_data(DATA_FILE)

    # Build features
    features_df = build_features(df)

    # Align close prices with features (features dropped some rows due to rolling)
    close_prices = df["close"].loc[features_df.index]

    # Create windowed samples
    X, y, paths = create_windows(features_df, close_prices, LOOKBACK, HORIZON)

    # Target distribution info
    tgt_up = np.sum(y > 0)
    tgt_down = np.sum(y < 0)
    tgt_flat = np.sum(y == 0)

    # Chronological split
    split_idx = int(len(y) * TRAIN_RATIO)
    X_train_full, X_test = X[:split_idx], X[split_idx:]
    y_train_full, y_test = y[:split_idx], y[split_idx:]
    test_paths = paths[split_idx:]  # bar-by-bar paths for SL/TP backtest

    # Validation split from training data
    val_split = int(len(X_train_full) * (1 - VAL_RATIO))
    X_train, X_val = X_train_full[:val_split], X_train_full[val_split:]
    y_train, y_val = y_train_full[:val_split], y_train_full[val_split:]

    num_features = X_train.shape[2]

    box("FEATURE ENGINEERING", [
        f"Features per bar:          {num_features}",
        f"Lookback window:           {LOOKBACK} bars",
        f"Prediction horizon:        {HORIZON} bars",
        f"Input shape:               ({LOOKBACK}, {num_features})",
        "---",
        f"Total samples:             {len(y):,}",
        f"  Target > 0 (up):         {tgt_up:,}  ({tgt_up/len(y)*100:.1f}%)",
        f"  Target < 0 (down):       {tgt_down:,}  ({tgt_down/len(y)*100:.1f}%)",
        f"  Target = 0 (flat):       {tgt_flat:,}  ({tgt_flat/len(y)*100:.1f}%)",
        f"  Target mean:             {np.mean(y):.4f} pts",
        f"  Target std:              {np.std(y):.4f} pts",
        "---",
        f"Train:                     {len(y_train):,}  ({len(y_train)/len(y)*100:.0f}%)",
        f"Validation:                {len(y_val):,}  ({len(y_val)/len(y)*100:.0f}%)",
        f"Test (holdout):            {len(y_test):,}  ({len(y_test)/len(y)*100:.0f}%)",
    ])

    # Normalize targets using training stats (train on normalized, backtest on real points)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std
    # y_test stays in real NQ points — predictions will be denormalized in backtest

    # Normalize features using training stats only
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
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Load saved weights if resuming
    resumed = False
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"\nERROR: Checkpoint not found: {args.resume}")
            sys.exit(1)
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        # Use saved normalization stats if available
        if "y_mean" in checkpoint and "y_std" in checkpoint:
            y_mean = checkpoint["y_mean"]
            y_std = checkpoint["y_std"]
        resumed = True
        resume_label = f"Resumed from:              {args.resume}"
    else:
        resume_label = f"Resumed from:              (none — training from scratch)"

    box("MODEL ARCHITECTURE", [
        f"Type:                      CNN + LSTM Hybrid",
        f"Conv layers:               2x Conv1D (64, 128)",
        f"LSTM:                      2 layers, 128 hidden",
        f"FC head:                   128 → 64 → 1",
        f"Total parameters:          {total_params:,}",
        f"Trainable parameters:      {trainable:,}",
        resume_label,
        "---",
        f"Optimizer:                 AdamW (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})",
        f"Loss:                      MSE",
        f"Max epochs:                {MAX_EPOCHS}",
        f"Early stop patience:       {EARLY_STOP_PATIENCE}",
        f"LR scheduler patience:     {LR_PATIENCE}",
        f"Gradient clipping:         {GRAD_CLIP}",
    ])

    # Train (unless backtest-only mode)
    history = None
    if args.backtest_only:
        if not resumed:
            print("\nERROR: --backtest-only requires --resume <checkpoint>")
            sys.exit(1)
        print("\nSkipping training (--backtest-only mode)")
    else:
        start_time = time.time()
        model, history = train_model(model, train_loader, val_loader, device, use_amp)
        elapsed = time.time() - start_time
        print(f"  Wall time: {elapsed / 60:.1f} minutes")

        # Save full checkpoint
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "num_features": num_features,
            "y_mean": y_mean,
            "y_std": y_std,
            "lookback": LOOKBACK,
            "horizon": HORIZON,
        }
        torch.save(checkpoint, CHECKPOINT_FILE)
        print(f"\nCheckpoint saved to {CHECKPOINT_FILE}")

    # Backtest (denormalize predictions back to real NQ points)
    backtest(model, test_loader, device, use_amp, y_mean=y_mean, y_std=y_std, history=history)

    # SL/TP backtest (if requested)
    if args.tp is not None or args.sl is not None:
        if args.tp is None or args.sl is None:
            print("\nERROR: --tp and --sl must both be specified together.")
            sys.exit(1)
        backtest_sltp(model, test_loader, device, use_amp,
                      y_mean=y_mean, y_std=y_std,
                      test_paths=test_paths,
                      tp_pct=args.tp, sl_pct=args.sl)


if __name__ == "__main__":
    main()
