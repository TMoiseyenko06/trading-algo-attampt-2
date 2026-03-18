"""
NQ Futures Live Trading Module
===============================
Fetches real-time 1-minute bars from TradingView, runs predictions through
the trained CNN+LSTM model, and manages trades with TP/SL/expiry logic.

Trade entry/exit is printed to CLI (manual execution).

Requires:
  - Trained checkpoint (nq_checkpoint.pt)
  - Training data (nq.dbn) for scaler calibration
  - TradingView credentials: TV_USERNAME / TV_PASSWORD env vars
  - pip install tvdatafeed
"""

import os
import sys
import time
import argparse
from datetime import datetime

from dotenv import load_dotenv

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from tvDatafeed import TvDatafeed, Interval

# Reuse model, features, and data loading from the training module
from nq_predictor import (
    NQPredictor,
    build_features,
    load_data,
    create_windows,
    LOOKBACK,
    HORIZON,
    TRADE_THRESHOLD,
    NQ_MULTIPLIER,
    CHECKPOINT_FILE,
    DATA_FILE,
    TRAIN_RATIO,
)

# How many raw bars to fetch from TradingView.
# build_features() needs ~60 bars of warm-up (SMA60 is the longest rolling window),
# plus LOOKBACK (90) bars of usable features = 150 minimum. Add buffer.
FETCH_BARS = 200


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def banner(text):
    w = 60
    print()
    print("=" * w)
    print(f"  {text}")
    print("=" * w)


# ─────────────────────────────────────────────
# Startup: load model + calibrate scaler
# ─────────────────────────────────────────────
def load_checkpoint(path, device):
    """Load trained model and normalization stats from checkpoint."""
    if not os.path.exists(path):
        print(f"ERROR: Checkpoint not found: {path}")
        sys.exit(1)

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = NQPredictor(ckpt["num_features"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, ckpt["y_mean"], ckpt["y_std"], ckpt["num_features"]


def calibrate_scaler(data_file, num_features):
    """Reproduce the exact StandardScaler from training by re-fitting on the
    training portion of the historical data (same logic as nq_predictor.py)."""
    if not os.path.exists(data_file):
        print(f"ERROR: Training data not found: {data_file}")
        print("The .dbn file is needed once at startup to calibrate the feature scaler.")
        sys.exit(1)

    print(f"  Loading {data_file} for scaler calibration...")
    df = load_data(data_file)
    features_df = build_features(df)
    close_prices = df["close"].loc[features_df.index]

    X, _, _ = create_windows(features_df, close_prices, LOOKBACK, HORIZON)
    split_idx = int(len(X) * TRAIN_RATIO)
    X_train = X[:split_idx]

    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, num_features))
    print(f"  Scaler calibrated on {len(X_train):,} training windows.")
    return scaler


# ─────────────────────────────────────────────
# TradingView data
# ─────────────────────────────────────────────
def connect_tv():
    """Connect to TradingView using credentials from environment."""
    username = os.environ.get("TV_USERNAME")
    password = os.environ.get("TV_PASSWORD")
    if not username or not password:
        print("ERROR: Set TV_USERNAME and TV_PASSWORD environment variables.")
        sys.exit(1)

    print(f"  Connecting to TradingView as {username}...")
    tv = TvDatafeed(username=username, password=password)
    return tv


def fetch_bars(tv, symbol, exchange, n_bars=FETCH_BARS):
    """Fetch recent 1-minute bars from TradingView."""
    df = tv.get_hist(
        symbol=symbol,
        exchange=exchange,
        interval=Interval.in_1_minute,
        n_bars=n_bars,
    )
    if df is None or df.empty:
        return None

    # tvDatafeed returns columns: open, high, low, close, volume
    # Ensure standard column names
    df.columns = [c.lower() for c in df.columns]
    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            print(f"  WARNING: Missing column '{col}' in TradingView data")
            return None

    return df[required].copy()


def fetch_latest_price(tv, symbol, exchange):
    """Fetch the most recent close price."""
    df = tv.get_hist(
        symbol=symbol,
        exchange=exchange,
        interval=Interval.in_1_minute,
        n_bars=2,
    )
    if df is None or df.empty:
        return None
    return float(df["close"].iloc[-1])


# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
def predict(model, device, scaler, y_mean, y_std, num_features, raw_df):
    """Build features from raw OHLCV, normalize, run model, return prediction in NQ points."""
    features_df = build_features(raw_df)

    if len(features_df) < LOOKBACK:
        return None, None

    # Take the last LOOKBACK rows as our input window
    window = features_df.iloc[-LOOKBACK:].values.astype(np.float32)

    # Normalize with the training scaler
    window_flat = window.reshape(-1, num_features)
    window_flat = scaler.transform(window_flat)
    np.nan_to_num(window_flat, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    window = window_flat.reshape(1, LOOKBACK, num_features)

    # Model forward pass
    X_tensor = torch.tensor(window, dtype=torch.float32).to(device)
    with torch.no_grad():
        raw_pred = model(X_tensor).item()

    # Denormalize to real NQ points
    pred_points = raw_pred * y_std + y_mean

    # Entry price is the last close in the raw data
    entry_price = float(raw_df["close"].iloc[-1])

    return pred_points, entry_price


# ─────────────────────────────────────────────
# Trade monitoring
# ─────────────────────────────────────────────
def monitor_trade(tv, symbol, exchange, direction, entry_price, tp_level, sl_level,
                  poll_interval):
    """Poll price until TP, SL, or expiry (HORIZON bars). Returns exit info dict."""
    bars_elapsed = 0
    last_bar_time = datetime.now()

    print()
    print("  --- Monitoring ---")

    while bars_elapsed < HORIZON:
        time.sleep(poll_interval)

        price = fetch_latest_price(tv, symbol, exchange)
        if price is None:
            print(f"  [{timestamp()}] Price fetch failed, retrying...")
            continue

        # Count new 1-minute bars by elapsed time
        now = datetime.now()
        elapsed_seconds = (now - last_bar_time).total_seconds()
        if elapsed_seconds >= 60:
            new_bars = int(elapsed_seconds // 60)
            bars_elapsed += new_bars
            last_bar_time = now

        move = price - entry_price
        move_in_direction = direction * move
        pnl_dollars = move * direction * NQ_MULTIPLIER

        bar_display = min(bars_elapsed, HORIZON)
        print(f"  Bar {bar_display:>2}/{HORIZON} | "
              f"Price: {price:,.2f} | "
              f"Move: {move:+.2f} pts | "
              f"P&L: ${pnl_dollars:+,.2f}")

        # Check TP
        if move_in_direction >= tp_level:
            return {
                "reason": "TP HIT",
                "exit_price": price,
                "pnl_points": tp_level,
                "pnl_dollars": tp_level * NQ_MULTIPLIER,
                "bars_held": bar_display,
            }

        # Check SL
        if move_in_direction <= -sl_level:
            return {
                "reason": "SL HIT",
                "exit_price": price,
                "pnl_points": -sl_level,
                "pnl_dollars": -sl_level * NQ_MULTIPLIER,
                "bars_held": bar_display,
            }

    # Expiry — fetch final price
    price = fetch_latest_price(tv, symbol, exchange)
    if price is None:
        price = entry_price  # fallback
    move = price - entry_price
    pnl_points = direction * move
    return {
        "reason": "EXPIRY",
        "exit_price": price,
        "pnl_points": pnl_points,
        "pnl_dollars": pnl_points * NQ_MULTIPLIER,
        "bars_held": HORIZON,
    }


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────
def run(args):
    load_dotenv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    banner("NQ LIVE TRADING MODULE")
    print(f"  Device:       {device}")
    print(f"  Checkpoint:   {args.checkpoint}")
    print(f"  Symbol:       {args.symbol} ({args.exchange})")
    print(f"  TP pct:       {args.tp_pct}")
    print(f"  SL pct:       {args.sl_pct}")
    print(f"  Threshold:    {args.threshold} pts")
    print(f"  Poll interval:{args.poll_interval}s")

    # Load model
    print()
    print("  Loading model...")
    model, y_mean, y_std, num_features = load_checkpoint(args.checkpoint, device)
    print(f"  Model loaded ({num_features} features, y_mean={y_mean:.4f}, y_std={y_std:.4f})")

    # Calibrate scaler
    scaler = calibrate_scaler(args.data_file, num_features)

    # Connect to TradingView
    tv = connect_tv()

    # Session stats
    total_trades = 0
    total_pnl = 0.0
    wins = 0

    banner("LIVE LOOP STARTED")
    print(f"  [{timestamp()}] Waiting for signals...\n")

    try:
        while True:
            # 1. Fetch recent bars
            raw_df = fetch_bars(tv, args.symbol, args.exchange)
            if raw_df is None:
                print(f"  [{timestamp()}] Data fetch failed, retrying in 60s...")
                time.sleep(60)
                continue

            # 2. Predict
            pred_points, entry_price = predict(
                model, device, scaler, y_mean, y_std, num_features, raw_df
            )
            if pred_points is None:
                print(f"  [{timestamp()}] Not enough feature data, retrying in 60s...")
                time.sleep(60)
                continue

            # 3. Check threshold
            if abs(pred_points) < args.threshold:
                print(f"  [{timestamp()}] Pred: {pred_points:+.2f} pts "
                      f"(below threshold {args.threshold}), skipping...")
                time.sleep(60)
                continue

            # 4. Trade setup
            direction = 1.0 if pred_points > 0 else -1.0
            direction_str = "LONG" if direction > 0 else "SHORT"
            tp_level = abs(pred_points) * args.tp_pct
            sl_level = tp_level * args.sl_pct

            tp_price = entry_price + direction * tp_level
            sl_price = entry_price - direction * sl_level

            # 5. Print trade entry
            banner(f"TRADE ENTRY — {direction_str}")
            print(f"  Time:          {timestamp()}")
            print(f"  Predicted move:{pred_points:+.2f} pts")
            print(f"  Direction:     {direction_str}")
            print(f"  Entry price:   {entry_price:,.2f}")
            print()
            print(f"  TP: {tp_level:+.2f} pts -> {tp_price:,.2f}  (tp_pct={args.tp_pct})")
            print(f"  SL: {-sl_level:+.2f} pts -> {sl_price:,.2f}  (sl_pct={args.sl_pct})")
            print(f"  Max bars:      {HORIZON}")

            # 6. Monitor trade
            result = monitor_trade(
                tv, args.symbol, args.exchange,
                direction, entry_price, tp_level, sl_level,
                args.poll_interval,
            )

            # 7. Print trade exit
            total_trades += 1
            total_pnl += result["pnl_dollars"]
            if result["pnl_dollars"] > 0:
                wins += 1

            banner(f"TRADE CLOSED — {result['reason']}")
            print(f"  Time:          {timestamp()}")
            print(f"  Exit reason:   {result['reason']} at bar {result['bars_held']}")
            print(f"  Exit price:    {result['exit_price']:,.2f}")
            print(f"  P&L:           {result['pnl_points']:+.2f} pts  (${result['pnl_dollars']:+,.2f})")
            print()
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            print(f"  Session:       {total_trades} trades | "
                  f"Win rate: {win_rate:.0f}% | "
                  f"Total P&L: ${total_pnl:+,.2f}")
            print()

            # 8. Brief pause before next cycle
            print(f"  [{timestamp()}] Waiting 60s before next scan...\n")
            time.sleep(60)

    except KeyboardInterrupt:
        banner("SESSION ENDED")
        if total_trades > 0:
            win_rate = wins / total_trades * 100
            print(f"  Trades:    {total_trades}")
            print(f"  Wins:      {wins} ({win_rate:.0f}%)")
            print(f"  Total P&L: ${total_pnl:+,.2f}")
        else:
            print("  No trades executed.")
        print()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="NQ Live Trading — CNN+LSTM predictions with TradingView data"
    )
    parser.add_argument("--tp-pct", type=float, default=1.0,
                        help="TP as fraction of predicted move (default: 1.0)")
    parser.add_argument("--sl-pct", type=float, default=0.5,
                        help="SL as fraction of TP (default: 0.5)")
    parser.add_argument("--threshold", type=float, default=TRADE_THRESHOLD,
                        help=f"Min predicted move to enter trade (default: {TRADE_THRESHOLD})")
    parser.add_argument("--poll-interval", type=int, default=10,
                        help="Seconds between price checks during trade (default: 10)")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_FILE,
                        help=f"Model checkpoint file (default: {CHECKPOINT_FILE})")
    parser.add_argument("--data-file", type=str, default=DATA_FILE,
                        help=f"Training data for scaler calibration (default: {DATA_FILE})")
    parser.add_argument("--symbol", type=str, default="NQ1!",
                        help="TradingView symbol (default: NQ1!)")
    parser.add_argument("--exchange", type=str, default="CME",
                        help="TradingView exchange (default: CME)")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
