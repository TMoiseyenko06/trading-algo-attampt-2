"""
NQ Futures Live Trading Module
===============================
Streams real-time 1-minute bars from Databento Live API, runs predictions
through the trained CNN+LSTM model, and manages trades with TP/SL/expiry logic.

Trade entry/exit is printed to CLI (manual execution).

Requires:
  - Trained checkpoint (nq_checkpoint.pt)
  - Training data (nq.dbn) for scaler calibration
  - Databento API key in .env: DATABENTO_API_KEY
  - pip install databento
"""

import os
import sys
import time
import logging
import argparse
import threading
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

from dotenv import load_dotenv

import numpy as np
import pandas as pd
import torch
import databento as db
from sklearn.preprocessing import StandardScaler

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

# How many raw bars to fetch for initial warmup.
# build_features() needs ~60 bars of warm-up (SMA60 is the longest rolling window),
# plus LOOKBACK (120) bars of usable features = 180 minimum. Add buffer.
FETCH_BARS = 200

# Max bars to keep in the rolling buffer (prevents unbounded memory growth)
MAX_BUFFER_BARS = 500

# Databento config
DATABENTO_DATASET = "GLBX.MDP3"
DATABENTO_SYMBOL = "NQ.n.0"  # front-month by open interest


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


def _get_api_key():
    """Read Databento API key from environment."""
    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        print("ERROR: Missing DATABENTO_API_KEY in .env file.")
        print("  Required variable:")
        print("    DATABENTO_API_KEY=db-your-api-key-here")
        sys.exit(1)
    return api_key


def _convert_price(price):
    """Convert Databento fixed-point price (1e-9 scale) to float."""
    p = float(price)
    if p > 1e6:
        p = p / 1e9
    return p


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
    training portion of the historical data (same logic as nq_predictor.py).

    Optimization: StandardScaler computes per-feature mean/std, which is
    identical whether computed on flattened windows or on the raw feature rows
    (since each row appears in exactly LOOKBACK consecutive windows in the
    training set, the mean/std are unchanged). This avoids the expensive
    create_windows() call on millions of rows."""
    if not os.path.exists(data_file):
        print(f"ERROR: Training data not found: {data_file}")
        print("The .dbn file is needed once at startup to calibrate the feature scaler.")
        sys.exit(1)

    print(f"  Loading {data_file} for scaler calibration...")
    df = load_data(data_file)
    features_df = build_features(df)

    # Determine training split the same way as nq_predictor.py:
    # number of possible windows = len(features_df) - LOOKBACK - HORIZON
    n_windows = len(features_df) - LOOKBACK - HORIZON
    split_idx = int(n_windows * TRAIN_RATIO)
    # Training windows use feature rows from index 0..split_idx+LOOKBACK-1
    train_end = split_idx + LOOKBACK
    train_features = features_df.iloc[:train_end].values.astype(np.float32)

    scaler = StandardScaler()
    scaler.fit(train_features)
    print(f"  Scaler calibrated on {len(train_features):,} feature rows "
          f"(~{split_idx:,} training windows).")
    return scaler


# ─────────────────────────────────────────────
# Databento Live streaming
# ─────────────────────────────────────────────
class LiveBarStream:
    """Streams 1-minute OHLCV bars from Databento Live API in a background thread.

    Maintains a rolling DataFrame of recent bars. The main thread can:
      - Wait for new bars via wait_for_bar()
      - Read the latest price via latest_price
      - Access the full rolling history via get_bars()
    """

    def __init__(self, api_key, warmup_bars=FETCH_BARS):
        self._api_key = api_key
        self._warmup_bars = warmup_bars
        self._lock = threading.Lock()
        self._new_bar_event = threading.Event()
        self._bars = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        self._latest_price = None
        self._live_client = None
        self._thread = None
        self._running = False
        self._error = None

    @property
    def latest_price(self):
        with self._lock:
            return self._latest_price

    def get_bars(self):
        """Return a copy of the rolling bar DataFrame."""
        with self._lock:
            return self._bars.copy()

    def bar_count(self):
        with self._lock:
            return len(self._bars)

    def wait_for_bar(self, timeout=120):
        """Block until a new bar arrives. Returns True if bar received, False on timeout."""
        self._new_bar_event.clear()
        return self._new_bar_event.wait(timeout=timeout)

    def start(self):
        """Fetch historical warmup bars, then start the live stream."""
        self._fetch_warmup()
        self._running = True
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the live stream."""
        self._running = False
        if self._live_client is not None:
            try:
                self._live_client.stop()
            except Exception:
                pass

    def _fetch_warmup(self):
        """Use Historical API to get initial bars for feature computation."""
        print(f"  Fetching {self._warmup_bars} warmup bars from Databento Historical...")
        client = db.Historical(self._api_key)
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=self._warmup_bars * 3)

        for attempt in range(1, 4):
            try:
                data = client.timeseries.get_range(
                    dataset=DATABENTO_DATASET,
                    symbols=DATABENTO_SYMBOL,
                    stype_in="continuous",
                    schema="ohlcv-1m",
                    start=start_time.strftime("%Y-%m-%dT%H:%M"),
                    end=end_time.strftime("%Y-%m-%dT%H:%M"),
                )

                df = data.to_df()
                if df.empty:
                    logger.warning(f"  Empty warmup data (attempt {attempt}/3)")
                    if attempt < 3:
                        time.sleep(2 ** attempt)
                    continue

                price_cols = ["open", "high", "low", "close"]
                for col in price_cols:
                    if col in df.columns and df[col].median() > 1e6:
                        df[col] = df[col] / 1e9

                if "volume" not in df.columns and "size" in df.columns:
                    df["volume"] = df["size"]

                keep_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
                df = df[keep_cols].copy()
                df = df.dropna()
                df = df[(df["close"] > 0) & (df["volume"] >= 0)]

                if len(df) > self._warmup_bars:
                    df = df.iloc[-self._warmup_bars:]

                df = df.reset_index(drop=True)

                with self._lock:
                    self._bars = df
                    self._latest_price = float(df["close"].iloc[-1])

                print(f"  Warmup complete: {len(df)} bars loaded, "
                      f"latest price: {self._latest_price:,.2f}")
                return

            except Exception as e:
                logger.warning(f"  Warmup fetch failed (attempt {attempt}/3): {e}")
                if attempt < 3:
                    time.sleep(2 ** attempt)

        print("ERROR: Failed to fetch warmup bars after 3 attempts.")
        sys.exit(1)

    def _stream_loop(self):
        """Background thread: connect to Databento Live and stream bars."""
        while self._running:
            try:
                self._live_client = db.Live(key=self._api_key)
                self._live_client.subscribe(
                    dataset=DATABENTO_DATASET,
                    schema="ohlcv-1m",
                    symbols=[DATABENTO_SYMBOL],
                    stype_in="continuous",
                )
                self._live_client.start()

                print(f"  [{timestamp()}] Live stream connected.")
                self._error = None

                for record in self._live_client:
                    if not self._running:
                        break
                    self._process_record(record)

            except Exception as e:
                if not self._running:
                    break
                self._error = str(e)
                logger.warning(f"  Live stream error: {e}")
                print(f"  [{timestamp()}] Live stream disconnected: {e}")
                print(f"  [{timestamp()}] Reconnecting in 5s...")
                time.sleep(5)

    def _process_record(self, record):
        """Extract OHLCV from a live record and append to rolling buffer."""
        # OhlcvMsg has open, high, low, close, volume attributes
        if not hasattr(record, "open"):
            return

        o = _convert_price(record.open)
        h = _convert_price(record.high)
        l = _convert_price(record.low)
        c = _convert_price(record.close)
        v = float(record.volume) if hasattr(record, "volume") else 0.0

        if c <= 0:
            return

        new_row = pd.DataFrame(
            [[o, h, l, c, v]],
            columns=["open", "high", "low", "close", "volume"],
        )

        with self._lock:
            self._bars = pd.concat([self._bars, new_row], ignore_index=True)
            # Trim to max buffer size
            if len(self._bars) > MAX_BUFFER_BARS:
                self._bars = self._bars.iloc[-MAX_BUFFER_BARS:].reset_index(drop=True)
            self._latest_price = c

        # Signal the main thread that a new bar arrived
        self._new_bar_event.set()


# ─────────────────────────────────────────────
# CSV fallback (unchanged)
# ─────────────────────────────────────────────
def load_csv(path):
    """Load OHLCV data from a TradingView-exported CSV file."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "volume" not in df.columns:
        for col in df.columns:
            if "vol" in col:
                df = df.rename(columns={col: "volume"})
                break
    if "volume" not in df.columns:
        print("  NOTE: No volume column found, filling with zeros")
        df["volume"] = 0.0
    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            print(f"ERROR: CSV missing required column '{col}'")
            print(f"  Found columns: {list(df.columns)}")
            sys.exit(1)
    return df[required].copy()


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
# Trade monitoring (now uses live stream)
# ─────────────────────────────────────────────
def monitor_trade(stream, direction, entry_price, tp_level, sl_level, poll_interval):
    """Monitor live price stream until TP, SL, or expiry (HORIZON bars). Returns exit_info dict."""
    bars_elapsed = 0
    last_bar_time = datetime.now()

    print()
    print("  --- Monitoring ---")

    while bars_elapsed < HORIZON:
        time.sleep(poll_interval)

        price = stream.latest_price
        if price is None:
            print(f"  [{timestamp()}] No price available, waiting...")
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

    # Expiry — use latest streamed price
    price = stream.latest_price
    if price is None:
        price = entry_price
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
    api_key = _get_api_key()

    banner("NQ LIVE TRADING MODULE")
    print(f"  Device:       {device}")
    print(f"  Checkpoint:   {args.checkpoint}")
    print(f"  Data source:  Databento Live ({DATABENTO_DATASET}, {DATABENTO_SYMBOL})")
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

    # CSV mode: single prediction, no Databento needed
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

    # Start live bar stream
    stream = LiveBarStream(api_key)
    stream.start()
    print(f"  Databento Live stream active.")

    # Session stats
    total_trades = 0
    total_pnl = 0.0
    wins = 0

    banner("LIVE LOOP STARTED — STREAMING")
    print(f"  [{timestamp()}] Waiting for new bars...\n")

    try:
        while True:
            # Block until a new 1-minute bar arrives from the live stream
            got_bar = stream.wait_for_bar(timeout=120)
            if not got_bar:
                print(f"  [{timestamp()}] No bar received in 120s, stream may be stale...")
                continue

            # Get rolling bar history
            raw_df = stream.get_bars()
            if len(raw_df) < LOOKBACK + 60:
                print(f"  [{timestamp()}] Buffering... {len(raw_df)} bars "
                      f"(need {LOOKBACK + 60})")
                continue

            # Predict
            pred_points, entry_price = predict(
                model, device, scaler, y_mean, y_std, num_features, raw_df
            )
            if pred_points is None:
                print(f"  [{timestamp()}] Not enough feature data, waiting...")
                continue

            # Check threshold
            if abs(pred_points) < args.threshold:
                print(f"  [{timestamp()}] Pred: {pred_points:+.2f} pts "
                      f"(below threshold {args.threshold}), skipping...")
                continue

            # Trade setup
            direction = 1.0 if pred_points > 0 else -1.0
            direction_str = "LONG" if direction > 0 else "SHORT"
            tp_level = abs(pred_points) * args.tp_pct
            sl_level = tp_level * args.sl_pct

            tp_price = entry_price + direction * tp_level
            sl_price = entry_price - direction * sl_level

            # Print trade entry
            banner(f"TRADE ENTRY — {direction_str}")
            print(f"  Time:          {timestamp()}")
            print(f"  Predicted move:{pred_points:+.2f} pts")
            print(f"  Direction:     {direction_str}")
            print(f"  Entry price:   {entry_price:,.2f}")
            print()
            print(f"  TP: {tp_level:+.2f} pts -> {tp_price:,.2f}  (tp_pct={args.tp_pct})")
            print(f"  SL: {-sl_level:+.2f} pts -> {sl_price:,.2f}  (sl_pct={args.sl_pct})")
            print(f"  Max bars:      {HORIZON}")

            # Monitor trade using live stream prices
            result = monitor_trade(
                stream, direction, entry_price, tp_level, sl_level,
                args.poll_interval,
            )

            # Print trade exit
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

    except KeyboardInterrupt:
        stream.stop()
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
        description="NQ Live Trading — CNN+LSTM predictions with Databento Live streaming"
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
                        help="Path to CSV file with OHLCV data (bypasses Databento)")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
