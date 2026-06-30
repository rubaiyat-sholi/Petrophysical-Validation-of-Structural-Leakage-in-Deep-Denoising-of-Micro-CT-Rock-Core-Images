"""
Training script for denoising models.

Usage
-----
py -3 train.py --model dncnn --epochs 50
py -3 train.py --model unet  --epochs 50
py -3 train.py --model unet_leakage --beta 0.1 --epochs 50

Output
------
runs/<model>_<timestamp>/
    best.pt          : best val-loss checkpoint
    last.pt          : last epoch checkpoint
    train_log.csv    : epoch-level metrics
"""

import argparse, os, time, csv
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime

from src.data   import make_loader
from src.models import DnCNN, UNet, UNetLeakage
from src.losses import StandardLoss, LeakageLoss


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default="unet_leakage",
                   choices=["dncnn", "unet", "unet_leakage"])
    p.add_argument("--epochs",  type=int,   default=50)
    p.add_argument("--batch",   type=int,   default=32)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--alpha",   type=float, default=0.5,
                   help="SSIM weight in standard loss")
    p.add_argument("--beta",    type=float, default=0.1,
                   help="Leakage penalty weight (unet_leakage only)")
    p.add_argument("--workers", type=int,   default=0,
                   help="DataLoader workers (0 = main process, safe on Windows)")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM  : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ── data ──────────────────────────────────────────────────────────────────
    train_loader = make_loader("train", batch_size=args.batch, num_workers=args.workers)
    val_loader   = make_loader("val",   batch_size=args.batch, num_workers=args.workers)
    print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    # ── model ──────────────────────────────────────────────────────────────────
    if args.model == "dncnn":
        model  = DnCNN().to(device)
        criterion = StandardLoss(alpha=args.alpha)
        use_leak  = False
    elif args.model == "unet":
        model  = UNet().to(device)
        criterion = StandardLoss(alpha=args.alpha)
        use_leak  = False
    else:  # unet_leakage
        model  = UNetLeakage().to(device)
        criterion = LeakageLoss(alpha=args.alpha, beta=args.beta)
        use_leak  = True

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}  |  Params: {n_params:,}")

    # ── optimiser ─────────────────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # ── output dir ────────────────────────────────────────────────────────────
    stamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    rundir = os.path.join("runs", f"{args.model}_{stamp}")
    os.makedirs(rundir, exist_ok=True)
    log_path = os.path.join(rundir, "train_log.csv")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["epoch", "train_loss", "val_loss"]
        if use_leak:
            header += ["train_std", "train_leak", "val_std", "val_leak"]
        writer.writerow(header)

    best_val = float("inf")

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_totals = {"total": 0.0, "std": 0.0, "leakage": 0.0}
        n_train = 0

        for lq, hq in train_loader:
            lq, hq = lq.to(device), hq.to(device)
            pred = model(lq)

            if use_leak:
                losses = criterion(pred, hq, lq)
                loss   = losses["total"]
                train_totals["std"]     += losses["std"].detach().item()
                train_totals["leakage"] += losses["leakage"].detach().item()
            else:
                loss = criterion(pred, hq)

            train_totals["total"] += loss.detach().item()
            n_train += 1

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # ── validation ────────────────────────────────────────────────────────
        model.eval()
        val_totals = {"total": 0.0, "std": 0.0, "leakage": 0.0}
        n_val = 0

        with torch.no_grad():
            for lq, hq in val_loader:
                lq, hq = lq.to(device), hq.to(device)
                pred = model(lq)

                if use_leak:
                    losses = criterion(pred, hq, lq)
                    val_totals["total"]   += losses["total"].item()
                    val_totals["std"]     += losses["std"].item()
                    val_totals["leakage"] += losses["leakage"].item()
                else:
                    val_totals["total"] += criterion(pred, hq).item()
                n_val += 1

        train_loss = train_totals["total"] / n_train
        val_loss   = val_totals["total"]   / n_val
        elapsed    = time.time() - t0

        # ── checkpoint ────────────────────────────────────────────────────────
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "val_loss": val_loss, "args": vars(args)},
                   os.path.join(rundir, "last.pt"))

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "val_loss": val_loss, "args": vars(args)},
                       os.path.join(rundir, "best.pt"))

        # ── log ───────────────────────────────────────────────────────────────
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            row = [epoch, f"{train_loss:.6f}", f"{val_loss:.6f}"]
            if use_leak:
                row += [
                    f"{train_totals['std']/n_train:.6f}",
                    f"{train_totals['leakage']/n_train:.6f}",
                    f"{val_totals['std']/n_val:.6f}",
                    f"{val_totals['leakage']/n_val:.6f}",
                ]
            writer.writerow(row)

        print(f"Ep {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.5f}  val={val_loss:.5f}  "
              f"best={best_val:.5f}  {elapsed:.1f}s")

    print(f"\nDone. Best val loss: {best_val:.6f}  -->  {rundir}/best.pt")


if __name__ == "__main__":
    main()
