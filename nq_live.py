"""
NQ Futures Live Trading Module
===============================
Streams real-time 1-minute bars from Databento Live API, trains a CNN+LSTM
model from scratch on live data, and manages trades with TP/SL/expiry logic.

The model starts with random weights and learns entirely from live market
data. A warmup phase collects samples before trading begins.

Trade entry/exit is printed to CLI (manual execution).

Requires:
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
import torch.nn as nn
import databento as db
from sklearn.preprocessing import StandardScaler
from collections import deque

# Reuse model and features from the training module
from nq_predictor import (
    NQPredictor,
    build_features,
    LOOKBACK,
    HORIZON,
    TRADE_THRESHOLD,
    NQ_MULTIPLIER,
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
# Online learning from live data (train from scratch)
# ─────────────────────────────────────────────
NUM_FEATURES = 17              # build_features() produces 17 columns
ONLINE_LR = 1e-3               # Higher LR for training from scratch
ONLINE_GRAD_CLIP = 1.0         # Gradient clipping
ONLINE_MIN_SAMPLES = 32        # Min samples before first training round
ONLINE_MAX_SAMPLES = 2048      # Max replay buffer size
ONLINE_TRAIN_EPOCHS = 5        # Gradient steps per training round
ONLINE_BATCH_SIZE = 16         # Mini-batch size
ONLINE_SAVE_INTERVAL = 10      # Save checkpoint every N training rounds
ONLINE_CHECKPOINT = "nq_live_model.pt"


class OnlineLearner:
    """Trains a CNN+LSTM model from scratch on live market data.

    Collects training samples passively from every bar:
      - At time T, snapshots the feature window (LOOKBACK bars)
      - At time T+HORIZON, records the actual price move as the target
      - Trains the model once enough samples accumulate

    The scaler and target normalization stats are built incrementally
    from live data — no pretrained model or historical data file needed.
    """

    def __init__(self, device):
        self._device = device
        self._num_features = NUM_FEATURES

        # Fresh random model
        self._model = NQPredictor(NUM_FEATURES)
        self._model.to(device)
        self._model.eval()

        # Incremental scaler — uses partial_fit to update with each new window
        self._scaler = StandardScaler()
        self._scaler_fitted = False
        self._scaler_samples = 0

        # Running target stats (for normalizing y values)
        self._y_sum = 0.0
        self._y_sum_sq = 0.0
        self._y_count = 0

        # Replay buffer: (X_window, y_target) tuples
        # X_window: raw features (LOOKBACK, NUM_FEATURES) before scaling
        self._buffer = deque(maxlen=ONLINE_MAX_SAMPLES)

        # Pending samples: waiting for HORIZON bars to pass
        # Each entry: (features_window, entry_close, bars_remaining)
        self._pending = []

        # Optimizer
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=ONLINE_LR, weight_decay=1e-4
        )
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer, mode="min", factor=0.5, patience=20
        )
        self._criterion = nn.HuberLoss(delta=1.0)

        self._train_rounds = 0
        self._total_samples = 0
        self._ready = False  # True once we have enough samples to trade

    @property
    def model(self):
        return self._model

    @property
    def scaler(self):
        return self._scaler

    @property
    def scaler_fitted(self):
        return self._scaler_fitted

    @property
    def y_mean(self):
        if self._y_count < 2:
            return 0.0
        return self._y_sum / self._y_count

    @property
    def y_std(self):
        if self._y_count < 2:
            return 1.0
        variance = (self._y_sum_sq / self._y_count) - (self.y_mean ** 2)
        return max(np.sqrt(max(variance, 0.0)), 1e-6)

    @property
    def ready(self):
        return self._ready

    @property
    def buffer_size(self):
        return len(self._buffer)

    @property
    def pending_count(self):
        return len(self._pending)

    def on_new_bar(self, raw_df):
        """Called every time a new bar arrives. Handles passive sample collection.

        1. Snapshots current feature window as a pending sample
        2. Decrements bars_remaining on all pending samples
        3. Completes any samples that have reached HORIZON bars
        4. Updates scaler incrementally
        5. Triggers training if enough samples
        """
        features_df = build_features(raw_df)
        if len(features_df) < LOOKBACK:
            return

        # Snapshot current window as a pending sample
        window = features_df.iloc[-LOOKBACK:].values.astype(np.float32)
        entry_close = float(raw_df["close"].iloc[-1])
        self._pending.append({
            "window": window,
            "entry_close": entry_close,
            "bars_remaining": HORIZON,
        })

        # Update scaler with this window's feature rows
        self._scaler.partial_fit(window)
        self._scaler_samples += len(window)
        self._scaler_fitted = True

        # Decrement and complete pending samples
        completed = []
        still_pending = []
        for sample in self._pending:
            sample["bars_remaining"] -= 1
            if sample["bars_remaining"] <= 0:
                # Target = current close - entry close
                actual_move = entry_close - sample["entry_close"]
                completed.append((sample["window"], actual_move))
            else:
                still_pending.append(sample)
        self._pending = still_pending

        # Add completed samples to replay buffer
        for window, target in completed:
            self._buffer.append((window, target))
            self._total_samples += 1
            # Update running target stats
            self._y_sum += target
            self._y_sum_sq += target * target
            self._y_count += 1

        # Train if we have enough samples
        buf_size = len(self._buffer)
        if buf_size >= ONLINE_MIN_SAMPLES and len(completed) > 0:
            self._train_step()

        # Mark ready once we've done at least one training round
        if self._train_rounds > 0 and not self._ready:
            self._ready = True
            print(f"  [Online] Model READY — {buf_size} samples, "
                  f"y_mean={self.y_mean:.2f}, y_std={self.y_std:.2f}")

    def record_trade_outcome(self, raw_df, actual_move):
        """Record an additional trade outcome (on top of passive collection)."""
        features_df = build_features(raw_df)
        if len(features_df) < LOOKBACK:
            return
        window = features_df.iloc[-LOOKBACK:].values.astype(np.float32)
        self._buffer.append((window, float(actual_move)))
        self._total_samples += 1
        self._y_sum += actual_move
        self._y_sum_sq += actual_move * actual_move
        self._y_count += 1

    def _train_step(self):
        """Run gradient steps on the replay buffer."""
        self._model.train()

        # Freeze batch norm during training from scratch too — let running
        # stats stabilize before using them
        for module in self._model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.eval()

        buf_list = list(self._buffer)
        n = len(buf_list)
        y_mean = self.y_mean
        y_std = self.y_std

        all_losses = []
        for epoch in range(ONLINE_TRAIN_EPOCHS):
            indices = np.random.permutation(n)

            for start in range(0, n, ONLINE_BATCH_SIZE):
                batch_idx = indices[start:start + ONLINE_BATCH_SIZE]
                if len(batch_idx) == 0:
                    break

                X_list = []
                y_list = []
                for i in batch_idx:
                    window, target = buf_list[i]
                    w = self._scaler.transform(window.reshape(-1, self._num_features))
                    np.nan_to_num(w, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                    X_list.append(w.reshape(LOOKBACK, self._num_features))
                    y_list.append((target - y_mean) / y_std)

                X_batch = torch.tensor(np.array(X_list), dtype=torch.float32).to(self._device)
                y_batch = torch.tensor(np.array(y_list), dtype=torch.float32).to(self._device)

                self._optimizer.zero_grad()
                preds = self._model(X_batch)
                loss = self._criterion(preds, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), ONLINE_GRAD_CLIP)
                self._optimizer.step()
                all_losses.append(loss.item())

        self._model.eval()
        self._train_rounds += 1

        avg_loss = np.mean(all_losses) if all_losses else 0
        self._scheduler.step(avg_loss)
        current_lr = self._optimizer.param_groups[0]["lr"]

        print(f"  [Online] Round {self._train_rounds}: "
              f"loss={avg_loss:.4f}, buffer={n}, "
              f"y_mean={y_mean:.2f}, y_std={y_std:.2f}, "
              f"lr={current_lr:.6f}")

        if self._train_rounds % ONLINE_SAVE_INTERVAL == 0:
            self._save_checkpoint()

    def _save_checkpoint(self):
        """Save the live-trained model."""
        torch.save({
            "model_state_dict": self._model.state_dict(),
            "num_features": self._num_features,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
            "lookback": LOOKBACK,
            "horizon": HORIZON,
            "scaler_mean": self._scaler.mean_.tolist() if self._scaler_fitted else None,
            "scaler_var": self._scaler.var_.tolist() if self._scaler_fitted else None,
            "scaler_n": int(self._scaler.n_samples_seen_) if self._scaler_fitted else 0,
            "online_train_rounds": self._train_rounds,
            "online_total_samples": self._total_samples,
            "y_sum": self._y_sum,
            "y_sum_sq": self._y_sum_sq,
            "y_count": self._y_count,
        }, ONLINE_CHECKPOINT)
        print(f"  [Online] Checkpoint saved: {ONLINE_CHECKPOINT} "
              f"(round {self._train_rounds})")

    def save_final(self):
        """Save final checkpoint when session ends."""
        if self._train_rounds > 0:
            self._save_checkpoint()
            print(f"  [Online] Final model saved after {self._train_rounds} rounds "
                  f"({self._total_samples} total samples).")

    @classmethod
    def from_checkpoint(cls, path, device):
        """Resume a live-trained model from a saved checkpoint."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        learner = cls(device)

        learner._model.load_state_dict(ckpt["model_state_dict"])
        learner._model.to(device)
        learner._model.eval()

        # Restore scaler
        if ckpt.get("scaler_mean") is not None:
            learner._scaler.mean_ = np.array(ckpt["scaler_mean"])
            learner._scaler.var_ = np.array(ckpt["scaler_var"])
            learner._scaler.scale_ = np.sqrt(learner._scaler.var_)
            learner._scaler.n_samples_seen_ = ckpt["scaler_n"]
            learner._scaler_fitted = True

        # Restore target stats
        learner._y_sum = ckpt.get("y_sum", 0.0)
        learner._y_sum_sq = ckpt.get("y_sum_sq", 0.0)
        learner._y_count = ckpt.get("y_count", 0)
        learner._train_rounds = ckpt.get("online_train_rounds", 0)
        learner._total_samples = ckpt.get("online_total_samples", 0)
        learner._ready = learner._train_rounds > 0

        # Recreate optimizer for the loaded model
        learner._optimizer = torch.optim.AdamW(
            learner._model.parameters(), lr=ONLINE_LR, weight_decay=1e-4
        )
        learner._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            learner._optimizer, mode="min", factor=0.5, patience=20
        )

        print(f"  [Online] Resumed from {path}: "
              f"{learner._train_rounds} rounds, "
              f"{learner._total_samples} samples, "
              f"y_mean={learner.y_mean:.2f}, y_std={learner.y_std:.2f}")
        return learner


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

        for attempt in range(1, 4):
            try:
                # Historical API has a delay — data isn't available right up to "now".
                # Compute end_time fresh each attempt so retries reflect current clock.
                # Use a generous 30-min offset to avoid edge cases.
                end_time = datetime.now(timezone.utc) - timedelta(minutes=30)
                start_time = end_time - timedelta(minutes=self._warmup_bars * 3)

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
def predict(learner, raw_df):
    """Build features from raw OHLCV, normalize, run model, return prediction in NQ points."""
    if not learner.ready or not learner.scaler_fitted:
        return None, None

    features_df = build_features(raw_df)
    if len(features_df) < LOOKBACK:
        return None, None

    window = features_df.iloc[-LOOKBACK:].values.astype(np.float32)

    # Normalize with the live-built scaler
    window_flat = window.reshape(-1, NUM_FEATURES)
    window_flat = learner.scaler.transform(window_flat)
    np.nan_to_num(window_flat, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    window = window_flat.reshape(1, LOOKBACK, NUM_FEATURES)

    X_tensor = torch.tensor(window, dtype=torch.float32).to(learner._device)
    with torch.no_grad():
        raw_pred = learner.model(X_tensor).item()

    # Denormalize to real NQ points
    pred_points = raw_pred * learner.y_std + learner.y_mean
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

    banner("NQ LIVE TRADING MODULE — LIVE LEARNING")
    print(f"  Device:       {device}")
    print(f"  Data source:  Databento Live ({DATABENTO_DATASET}, {DATABENTO_SYMBOL})")
    print(f"  TP pct:       {args.tp_pct}")
    print(f"  SL pct:       {args.sl_pct}")
    print(f"  Threshold:    {args.threshold} pts")
    print(f"  Poll interval:{args.poll_interval}s")
    print(f"  Min samples:  {ONLINE_MIN_SAMPLES} (before trading)")

    # Initialize or resume learner
    print()
    if args.resume and os.path.exists(args.resume):
        print(f"  Resuming from checkpoint: {args.resume}")
        learner = OnlineLearner.from_checkpoint(args.resume, device)
    else:
        print("  Initializing fresh model (random weights)...")
        learner = OnlineLearner(device)
        print(f"  Model created ({NUM_FEATURES} features, {ONLINE_MAX_SAMPLES} max buffer)")

    # Start live bar stream
    stream = LiveBarStream(api_key)
    stream.start()
    print(f"  Databento Live stream active.")

    # Session stats
    total_trades = 0
    total_pnl = 0.0
    wins = 0

    banner("LIVE LOOP — COLLECTING & LEARNING")
    if not learner.ready:
        print(f"  [{timestamp()}] Warmup phase: collecting {ONLINE_MIN_SAMPLES}+ samples "
              f"before trading...")
        print(f"  [{timestamp()}] Each sample takes {HORIZON} bars ({HORIZON} min) to complete.")
    print(f"  [{timestamp()}] Waiting for new bars...\n")

    try:
        while True:
            got_bar = stream.wait_for_bar(timeout=120)
            if not got_bar:
                print(f"  [{timestamp()}] No bar received in 120s, stream may be stale...")
                continue

            raw_df = stream.get_bars()
            if len(raw_df) < LOOKBACK + 60:
                print(f"  [{timestamp()}] Buffering... {len(raw_df)} bars "
                      f"(need {LOOKBACK + 60})")
                continue

            # Feed every bar to the learner for passive sample collection
            learner.on_new_bar(raw_df)

            # During warmup, just show collection progress
            if not learner.ready:
                print(f"  [{timestamp()}] Collecting: "
                      f"{learner.buffer_size}/{ONLINE_MIN_SAMPLES} samples, "
                      f"{learner.pending_count} pending")
                continue

            # Predict
            pred_points, entry_price = predict(learner, raw_df)
            if pred_points is None:
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

            banner(f"TRADE ENTRY — {direction_str}")
            print(f"  Time:          {timestamp()}")
            print(f"  Predicted move:{pred_points:+.2f} pts")
            print(f"  Direction:     {direction_str}")
            print(f"  Entry price:   {entry_price:,.2f}")
            print()
            print(f"  TP: {tp_level:+.2f} pts -> {tp_price:,.2f}  (tp_pct={args.tp_pct})")
            print(f"  SL: {-sl_level:+.2f} pts -> {sl_price:,.2f}  (sl_pct={args.sl_pct})")
            print(f"  Max bars:      {HORIZON}")

            entry_bars_snapshot = raw_df.copy()

            result = monitor_trade(
                stream, direction, entry_price, tp_level, sl_level,
                args.poll_interval,
            )

            total_trades += 1
            total_pnl += result["pnl_dollars"]
            if result["reason"] == "TP HIT":
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

            # Record trade outcome as an extra training sample
            actual_move = result["exit_price"] - entry_price
            learner.record_trade_outcome(entry_bars_snapshot, actual_move)
            print(f"  [Online] Trade outcome: predicted={pred_points:+.2f}, "
                  f"actual={actual_move:+.2f} pts")

    except KeyboardInterrupt:
        stream.stop()
        learner.save_final()
        banner("SESSION ENDED")
        if total_trades > 0:
            win_rate = wins / total_trades * 100
            print(f"  Trades:    {total_trades}")
            print(f"  Wins:      {wins} ({win_rate:.0f}%)")
            print(f"  Total P&L: ${total_pnl:+,.2f}")
        print(f"  Online:    {learner._train_rounds} training rounds, "
              f"{learner._total_samples} samples learned")
        print()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="NQ Live Trading — trains CNN+LSTM from scratch on live data"
    )
    parser.add_argument("--tp-pct", type=float, default=1.0,
                        help="TP as fraction of predicted move (default: 1.0)")
    parser.add_argument("--sl-pct", type=float, default=0.5,
                        help="SL as fraction of TP (default: 0.5)")
    parser.add_argument("--threshold", type=float, default=TRADE_THRESHOLD,
                        help=f"Min predicted move to enter trade (default: {TRADE_THRESHOLD})")
    parser.add_argument("--poll-interval", type=int, default=10,
                        help="Seconds between price checks during trade (default: 10)")
    parser.add_argument("--resume", type=str, default=ONLINE_CHECKPOINT,
                        help=f"Resume from a saved live checkpoint (default: {ONLINE_CHECKPOINT})")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
