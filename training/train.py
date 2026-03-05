"""
FarmGuard AI — Training Script
EfficientNet-B4 fine-tuned on PlantVillage dataset

Usage:
    python train.py --data_dir /path/to/PlantVillage --epochs 30 --batch_size 32

PlantVillage Dataset:
    Download from: https://www.kaggle.com/datasets/emmarex/plantdisease
    Or: https://github.com/spMohanty/PlantVillage-Dataset
    
    Expected directory structure:
    data_dir/
        Apple___Apple_scab/
            image1.jpg
            image2.jpg
        Apple___Black_rot/
            ...
        (38 class folders total)
"""

import os
import argparse
import time
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from app.models.classifier import build_model, get_training_transforms, get_inference_transforms, NUM_CLASSES


def parse_args():
    parser = argparse.ArgumentParser(description="Train FarmGuard AI EfficientNet-B4")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to PlantVillage dataset root")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Where to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=30, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--val_split", type=float, default=0.15, help="Fraction of data for validation")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--freeze_epochs", type=int, default=5,
                        help="Epochs to train only the head (backbone frozen)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--amp", action="store_true", help="Use Automatic Mixed Precision (faster on modern GPUs)")
    return parser.parse_args()


def load_datasets(data_dir: str, val_split: float):
    """Load and split PlantVillage dataset."""
    full_dataset = datasets.ImageFolder(
        root=data_dir,
        transform=get_training_transforms()
    )

    n_val = int(len(full_dataset) * val_split)
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # Validation uses inference transforms (no augmentation)
    val_ds.dataset = copy.deepcopy(full_dataset)
    val_ds.dataset.transform = get_inference_transforms()

    print(f"  Train samples: {n_train:,}")
    print(f"  Val samples:   {n_val:,}")
    print(f"  Classes found: {len(full_dataset.classes)}")

    if len(full_dataset.classes) != NUM_CLASSES:
        print(f"⚠️  WARNING: Found {len(full_dataset.classes)} classes, expected {NUM_CLASSES}")
        print("   Check your dataset directory structure.")

    return train_ds, val_ds, full_dataset.classes


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 50 == 0:
            print(f"    [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {running_loss/total:.4f} | "
                  f"Acc: {100.*correct/total:.1f}%")

    return running_loss / total, 100. * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100. * correct / total


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🌱 FarmGuard AI Training")
    print(f"   Device: {device}")
    print(f"   Data:   {args.data_dir}")
    print(f"   Epochs: {args.epochs} (freeze backbone for first {args.freeze_epochs})\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds, val_ds, class_names = load_datasets(args.data_dir, args.val_split)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)

    # Save class-to-index mapping alongside model
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    torch.save(class_to_idx, os.path.join(args.output_dir, "class_to_idx.pt"))

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"  Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.2f}%")

    # ── Training Strategy: Two-Phase ──────────────────────────────────────────
    # Phase 1 (freeze_epochs): Only train the new head — fast convergence
    # Phase 2 (remaining epochs): Unfreeze full backbone with lower LR — fine-tune

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for better generalization
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device == "cuda") else None

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Phase switching
        if epoch < args.freeze_epochs:
            # Freeze backbone, only train classifier head
            for param in model.features.parameters():
                param.requires_grad = False
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr, weight_decay=1e-4
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=args.freeze_epochs)
            phase = "HEAD ONLY"
        elif epoch == args.freeze_epochs:
            # Unfreeze backbone, use lower LR for fine-tuning
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW([
                {"params": model.features.parameters(), "lr": args.lr * 0.1},
                {"params": model.classifier.parameters(), "lr": args.lr}
            ], weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.freeze_epochs)
            phase = "FULL"
        else:
            phase = "FULL"

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"\nEpoch [{epoch+1}/{args.epochs}] ({phase}) — {elapsed:.0f}s")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(args.output_dir, "farmguard_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "best_val_acc": best_val_acc,
                "class_names": class_names,
                "args": vars(args)
            }, best_path)
            print(f"  ✅ New best! Saved to {best_path}")

        # Save latest checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(args.output_dir, f"farmguard_epoch{epoch+1}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "best_val_acc": best_val_acc,
                "history": history,
                "class_names": class_names,
            }, ckpt_path)

    print(f"\n🎉 Training complete! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"   Best model saved to: {os.path.join(args.output_dir, 'farmguard_best.pt')}")
    return history


if __name__ == "__main__":
    main()
