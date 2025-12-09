"""Train a simple classifier on DiffusionFER cropped images and export to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path
from collections import Counter
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, models, transforms


LABELS: List[str] = [
    "HighNegative",
    "MediumNegative",
    "LowNegative",
    "neutral",
    "MediumPositive",
    "HighPositive",
]


def make_loaders(data_dir: Path, batch_size: int, val_split: float):
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    full_ds = datasets.ImageFolder(root=str(data_dir), transform=tfm)
    val_len = int(len(full_ds) * val_split)
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len])
    # Class-balanced sampler to reduce happy bias.
    class_counts = Counter(full_ds.targets)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    train_indices = train_ds.indices  # type: ignore[attr-defined]
    sample_weights = [class_weights[full_ds.targets[i]] for i in train_indices]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, full_ds.classes


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Train MultiEmoVA classifier and export ONNX.")
    parser.add_argument("--data", type=Path, default=Path("MultiEmoVA"), help="Path to class folders.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--onnx-out", type=Path, default=Path("models/affect.onnx"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.onnx_out.parent.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, classes = make_loaders(args.data, args.batch_size, args.val_split)
    if classes != LABELS:
        print(f"Warning: dataset classes {classes} differ from expected label order {LABELS}")

    device = torch.device(args.device)
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}: train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")

    model.eval()
    dummy = torch.randn(1, 3, 224, 224, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(args.onnx_out),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"Exported ONNX to {args.onnx_out}")
    print(f"Label order: {classes}")


if __name__ == "__main__":
    main()
