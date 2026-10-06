"""Обратна връзка от потребителя и дообучение (fine-tuning) на модела.

Примерите се пазят като 28x28 PNG в feedback/<цифра>/ - точно каквото вижда моделът.
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from model import get_device
from train import (BASE_DIR, BEST_MODEL_PATH, DATA_DIR, MNIST_MEAN, MNIST_STD, MODELS_DIR,
                   evaluate, get_dataloaders, get_transform, train_one_epoch)
from utils import load_model

FEEDBACK_DIR = BASE_DIR / "feedback"
FINETUNED_PATH = MODELS_DIR / "finetuned_model.pt"
BACKUP_PATH = MODELS_DIR / "best_model.backup.pt"

MIN_FOR_HOLDOUT = 20   # под толкова примери няма смисъл да държим настрана част
MAX_REPEAT = 50        # най-много колко пъти повтаряме един пример


def save_sample(pixels, label):
    """Записва обработена 28x28 картинка (float 0..1, бяла цифра на черно) към класа label."""
    folder = FEEDBACK_DIR / str(int(label))
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000:06d}.png"
    Image.fromarray((np.asarray(pixels) * 255).round().astype(np.uint8)).save(folder / name)


def count_samples():
    """Речник {цифра: брой запазени примери}."""
    return {d: len(list((FEEDBACK_DIR / str(d)).glob("*.png"))) for d in range(10)}


def load_samples():
    """Връща (тензор N x 1 x 28 x 28 нормализиран, етикети N) или (None, None), ако няма примери."""
    xs, ys = [], []
    for digit in range(10):
        for path in sorted((FEEDBACK_DIR / str(digit)).glob("*.png")):
            pixels = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
            xs.append((pixels - MNIST_MEAN) / MNIST_STD)
            ys.append(digit)
    if not xs:
        return None, None
    return torch.from_numpy(np.stack(xs)).unsqueeze(1), torch.tensor(ys)


class _Samples(Dataset):
    """Като TensorDataset, но етикетът е обикновено число (както в MNIST), та да се смесват."""

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], int(self.y[i])


@torch.no_grad()
def _accuracy(model, x, y, device):
    model.eval()
    preds = model(x.to(device)).argmax(dim=1).cpu()
    return (preds == y).float().mean().item()


def _metrics(model, test_loader, criterion, device, x_train, y_train, x_hold, y_hold):
    return {
        "mnist_test": evaluate(model, test_loader, criterion, device)[1],
        "feedback_train": _accuracy(model, x_train, y_train, device),
        "feedback_holdout": _accuracy(model, x_hold, y_hold, device) if len(y_hold) else None,
    }


def fine_tune(epochs=5, lr=1e-4, replay=2000, batch_size=64, seed=0, save_path=FINETUNED_PATH):
    """Дообучава best_model.pt върху запазените примери + replay от MNIST.

    Връща речник с метрики преди/след. Записва резултата в save_path (не презаписва best_model).
    """
    x, y = load_samples()
    if x is None:
        raise ValueError("Няма запазени примери. Запази поне един от страница Inference.")

    device = get_device()
    generator = torch.Generator().manual_seed(seed)

    # Част от примерите се държат настрана, за да има честна оценка (ако са достатъчно).
    order = torch.randperm(len(y), generator=generator)
    n_hold = len(y) // 5 if len(y) >= MIN_FOR_HOLDOUT else 0
    hold_idx, train_idx = order[:n_hold], order[n_hold:]
    x_train, y_train = x[train_idx], y[train_idx]
    x_hold, y_hold = x[hold_idx], y[hold_idx]

    criterion = nn.CrossEntropyLoss()
    test_loader = get_dataloaders(256)[2]
    before = _metrics(load_model(BEST_MODEL_PATH, device), test_loader, criterion, device,
                      x_train, y_train, x_hold, y_hold)

    # Replay: случайни стари MNIST картинки, за да не "забрави" модела останалото.
    from torchvision import datasets
    full = datasets.MNIST(DATA_DIR, train=True, download=True, transform=get_transform())
    replay_ds = Subset(full, torch.randperm(len(full), generator=generator)[:replay].tolist())

    # Oversampling: примерите са малко, затова ги повтаряме, за да не се загубят сред replay.
    repeat = max(1, min(MAX_REPEAT, replay // (2 * len(y_train))))
    feedback_ds = ConcatDataset([_Samples(x_train, y_train)] * repeat)
    loader = DataLoader(ConcatDataset([replay_ds, feedback_ds]), batch_size=batch_size, shuffle=True)

    model = load_model(BEST_MODEL_PATH, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        train_one_epoch(model, loader, criterion, optimizer, device)

    after = _metrics(model, test_loader, criterion, device, x_train, y_train, x_hold, y_hold)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)

    return {"before": before, "after": after, "n_samples": len(y), "n_train": len(y_train),
            "n_holdout": n_hold, "repeat": repeat, "replay": replay}
