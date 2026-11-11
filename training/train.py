"""Train the volatility model. Run from the project root:  python -m training.train"""
import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn

from app.data import ANN, N_FEATURES, WINDOW, history, make_features
from app.model import VolGRU

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "XOM", "JNJ",
           "WMT", "KO", "DIS", "NFLX", "AMD", "BA", "GS", "PFE", "INTC", "SPY"]
HORIZON = 5
EPOCHS = 25


def windows(df):
    f = make_features(df)
    r = np.log(df["Close"].astype(float)).diff()
    f["y"] = r.rolling(HORIZON).std().shift(-HORIZON) * ANN   # target: realized vol over the NEXT 5 days
    a = f.dropna().to_numpy(np.float32)
    if len(a) < WINDOW + 100:
        return None
    X = sliding_window_view(a[:, :N_FEATURES], WINDOW, axis=0).transpose(0, 2, 1).copy()
    return X, a[WINDOW - 1:, N_FEATURES]


def main():
    parts = {"tr": ([], []), "va": ([], [])}
    for t in TICKERS:
        try:
            df = history(t, "10y")
        except Exception as e:
            print("failed", t, e)
            continue
        w = windows(df) if len(df) else None
        if w is None:
            print("skipped", t)
            continue
        X, y = w
        cut = int(len(X) * 0.8)                      # chronological split: no peeking into the future
        parts["tr"][0].append(X[:cut]); parts["tr"][1].append(y[:cut])
        parts["va"][0].append(X[cut:]); parts["va"][1].append(y[cut:])
        print("loaded", t, len(X))
    if len(parts["tr"][0]) < 5:
        raise SystemExit("Could not download enough price history. Check your internet connection and run again.")
    Xtr, ytr = (np.concatenate(p) for p in parts["tr"])
    Xva, yva = (np.concatenate(p) for p in parts["va"])
    mean, std = Xtr.reshape(-1, N_FEATURES).mean(0), Xtr.reshape(-1, N_FEATURES).std(0) + 1e-8
    norm = lambda X: torch.from_numpy((X - mean) / std).float()
    Xt, yt, Xv, yv = norm(Xtr), torch.from_numpy(ytr), norm(Xva), torch.from_numpy(yva)

    net, loss_fn = VolGRU(), nn.HuberLoss(delta=0.1)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    best, state = float("inf"), None
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss_fn(net(Xt[idx]), yt[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            mae = (net(Xv) - yv).abs().mean().item()
        print(f"epoch {ep + 1:02d}  val MAE {mae:.4f}")
        if mae < best:
            best, state = mae, {k: v.clone() for k, v in net.state_dict().items()}

    base = float(np.abs(Xva[:, -1, 4] * ANN - yva).mean())   # 20-day trailing-vol benchmark
    print(f"\nModel MAE {best:.4f}  vs  20d-vol baseline MAE {base:.4f}  ({(1 - best / base) * 100:+.1f}% better)")
    torch.save({"state_dict": state, "mean": mean.tolist(), "std": std.tolist(),
                "metrics": {"val_mae": round(best, 4), "baseline_mae": round(base, 4),
                            "horizon_days": HORIZON, "train_windows": int(len(Xt))}}, "model.pt")
    print("saved model.pt")


if __name__ == "__main__":
    main()
