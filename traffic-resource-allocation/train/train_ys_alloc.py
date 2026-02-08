import os
import sys

# ✅ 실행 위치/환경에 관계없이 루트 import 되게 안전장치
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import torch
from torch.utils.data import DataLoader

import config
from utils.dataset import TrafficAllocDataset
from utils.metrics import kl_div, mse, mae, jain_fairness, max_share
from models.ys_alloc_net import YSAllocNet
from models.rule_based import proportional_last_step

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_mode", choices=["next_step", "last_step"], default="next_step")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    args = ap.parse_args()

    train_npz = os.path.join(config.PROCESSED_DATA_DIR, "traffic_data_train.npz")
    test_npz  = os.path.join(config.PROCESSED_DATA_DIR, "traffic_data_test.npz")

    train_ds = TrafficAllocDataset(train_npz, label_mode=args.label_mode)
    test_ds  = TrafficAllocDataset(test_npz,  label_mode=args.label_mode)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YSAllocNet(window_size=config.WINDOW_SIZE, hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(os.path.join(PROJECT_ROOT, "artifacts"), exist_ok=True)
    best = float("inf")

    # -----------------------------
    # Rule-based baseline on test
    # -----------------------------
    rb = {"KL":0, "MSE":0, "MAE":0, "Jain":0, "MaxShare":0}
    with torch.no_grad():
        kls=[]; mses=[]; maes=[]; js=[]; mxs=[]
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = proportional_last_step(x)
            kls.append(kl_div(y, pred).item())
            mses.append(mse(y, pred).item())
            maes.append(mae(y, pred).item())
            js.append(jain_fairness(pred).item())
            mxs.append(max_share(pred).item())
        rb = {"KL":sum(kls)/len(kls), "MSE":sum(mses)/len(mses), "MAE":sum(maes)/len(maes),
              "Jain":sum(js)/len(js), "MaxShare":sum(mxs)/len(mxs)}

    print(f"\n[Rule-based] proportional_last_step | label_mode={args.label_mode}")
    for k,v in rb.items():
        print(f"  {k:8s}: {v:.6f}")

    # -----------------------------
    # Train
    # -----------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = kl_div(y, pred)

            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item()

        tr_loss /= max(len(train_loader), 1)

        # eval
        model.eval()
        te_loss = 0.0
        te_mse = te_mae = te_jain = te_mx = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                te_loss += kl_div(y, pred).item()
                te_mse  += mse(y, pred).item()
                te_mae  += mae(y, pred).item()
                te_jain += jain_fairness(pred).item()
                te_mx   += max_share(pred).item()

        n = max(len(test_loader), 1)
        te_loss /= n; te_mse /= n; te_mae /= n; te_jain /= n; te_mx /= n

        if te_loss < best:
            best = te_loss
            torch.save(model.state_dict(), os.path.join(PROJECT_ROOT, "artifacts", "ys_alloc_net.pt"))

        print(f"\n[Epoch {epoch:02d}] train_KL={tr_loss:.6f} | test_KL={te_loss:.6f} (best={best:.6f})")
        print(f"  test_MSE={te_mse:.6f}  test_MAE={te_mae:.6f}  Jain={te_jain:.6f}  MaxShare={te_mx:.6f}")

    print("\n✅ saved:", os.path.join("artifacts", "ys_alloc_net.pt"))

if __name__ == "__main__":
    main()
