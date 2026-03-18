"""
NQ Futures Live Trading Module
===============================
Fetches real-time 1-minute bars from TradingView, runs predictions through
the trained CNN+LSTM model, and manages trades with TP/SL/expiry logic.

Trade entry/exit is printed to CLI (manual execution).

Requires:
  - Trained checkpoint (nq_checkpoint.pt)
  - Training data (nq.dbn) for scaler calibration
  - TradingView session token: TV_AUTH_TOKEN in .env
  - pip install tvdatafeed
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime

logger = logging.getLogger(__name__)

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
def _make_tv():
    """Create a fresh TvDatafeed instance with the session token and a longer
    websocket timeout (the default 5 s is too aggressive)."""
    token = os.environ.get("TV_AUTH_TOKEN")
    if not token:
        print("ERROR: Set TV_AUTH_TOKEN in your .env file.")
        print("  1. Log into tradingview.com in your browser")
        print("  2. DevTools (F12) → Application → Cookies → sessionid")
        print("  3. Copy the value into .env as TV_AUTH_TOKEN=<value>")
        sys.exit(1)

    tv = TvDatafeed()
    tv.token = token
    # Bump the websocket timeout from the default 5 s to 30 s
    TvDatafeed._TvDatafeed__ws_timeout = 30
    return tv


def connect_tv():
    """Connect to TradingView using session token from environment."""
    print("  Connecting to TradingView with session token...")
    tv = _make_tv()
    return tv


def _tv_call(tv, fn, max_retries=3, **kwargs):
    """Call a TvDatafeed method with automatic reconnect on failure.

    If the websocket drops (``Connection to remote host was lost``), we build
    a brand-new TvDatafeed instance and retry up to *max_retries* times with
    exponential back-off (2 s, 4 s, 8 s).
    """
    for attempt in range(1, max_retries + 1):
        try:
            result = fn(**kwargs)
            return tv, result
        except Exception as e:
            logger.warning(f"  TV call failed (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  Reconnecting in {wait}s...")
                time.sleep(wait)
                tv = _make_tv()
            else:
                print(f"  TV call failed after {max_retries} attempts.")
                return tv, None


def fetch_bars(tv, symbol, exchange, n_bars=FETCH_BARS):
    """Fetch recent 1-minute bars from TradingView (with auto-reconnect)."""
    tv, df = _tv_call(
        tv,
        tv.get_hist,
        symbol=symbol,
        exchange=exchange,
        interval=Interval.in_1_minute,
        n_bars=n_bars,
    )
    if df is None or (hasattr(df, "empty") and df.empty):
        return tv, None

    # tvDatafeed returns columns: open, high, low, close, volume
    df.columns = [c.lower() for c in df.columns]
    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            print(f"  WARNING: Missing column '{col}' in TradingView data")
            return tv, None

    return tv, df[required].copy()


def load_csv(path):
    """Load OHLCV data from a TradingView-exported CSV file."""
    import pandas as pd

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    # TV exports 'volume' or 'Volume' — normalize
    if "volume" not in df.columns:
        for col in df.columns:
            if "vol" in col:
                df = df.rename(columns={col: "volume"})
                break
    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            print(f"ERROR: CSV missing required column '{col}'")
            print(f"  Found columns: {list(df.columns)}")
            sys.exit(1)
    return df[required].copy()


def fetch_latest_price(tv, symbol, exchange):
    """Fetch the most recent close price (with auto-reconnect)."""
    tv, df = _tv_call(
        tv,
        tv.get_hist,
        symbol=symbol,
        exchange=exchange,
        interval=Interval.in_1_minute,
        n_bars=2,
    )
    if df is None or (hasattr(df, "empty") and df.empty):
        return tv, None
    return tv, float(df["close"].iloc[-1])


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
    """Poll price until TP, SL, or expiry (HORIZON bars). Returns (tv, exit_info)."""
    bars_elapsed = 0
    last_bar_time = datetime.now()

    print()
    print("  --- Monitoring ---")

    while bars_elapsed < HORIZON:
        time.sleep(poll_interval)

        tv, price = fetch_latest_price(tv, symbol, exchange)
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
            return tv, {
                "reason": "TP HIT",
                "exit_price": price,
                "pnl_points": tp_level,
                "pnl_dollars": tp_level * NQ_MULTIPLIER,
                "bars_held": bar_display,
            }

        # Check SL
        if move_in_direction <= -sl_level:
            return tv, {
                "reason": "SL HIT",
                "exit_price": price,
                "pnl_points": -sl_level,
                "pnl_dollars": -sl_level * NQ_MULTIPLIER,
                "bars_held": bar_display,
            }

    # Expiry — fetch final price
    tv, price = fetch_latest_price(tv, symbol, exchange)
    if price is None:
        price = entry_price  # fallback
    move = price - entry_price
    pnl_points = direction * move
    return tv, {
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

    # CSV mode: single prediction, no TradingView needed
    if args.csv:
        banner("NQ PREDICTION — CSV MODE")
        raw_df = load_csv(args.csv)
        print(f"  Loaded {len(raw_df)} bars from {args.csv}")

        pred_points, entry_price = predict(
            model, device, scaler, y_mean, y_std, num_features, raw_df
        )
        if pred_points is None:
            print("  ERROR: Not enough data for prediction (need ~150+ bars)")
            sys.exit(1)

        direction = "LONG" if pred_points > 0 else "SHORT"
        tp = abs(pred_points) * args.tp_pct
        sl = tp * args.sl_pct
        entry = entry_price

        print(f"  Predicted move: {pred_points:+.2f} pts")
        print(f"  Direction:      {direction}")
        print(f"  Entry price:    {entry:,.2f}")
        print(f"  TP: {tp:+.2f} pts -> {entry + (1 if pred_points > 0 else -1) * tp:,.2f}")
        print(f"  SL: {-sl:+.2f} pts -> {entry - (1 if pred_points > 0 else -1) * sl:,.2f}")
        print(f"  Threshold:      {args.threshold} pts")
        if abs(pred_points) < args.threshold:
            print(f"  ** Below threshold — would NOT trade **")
        else:
            print(f"  ** Above threshold — would ENTER {direction} **")
        return

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
            tv, raw_df = fetch_bars(tv, args.symbol, args.exchange)
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
            tv, result = monitor_trade(
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
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to CSV file with OHLCV data (bypasses TradingView)")
    parser.add_argument("--symbol", type=str, default="NQ1!",
                        help="TradingView symbol (default: NQ1!)")
    parser.add_argument("--exchange", type=str, default="CME",
                        help="TradingView exchange (default: CME)")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
