"""Обучение на MNISTNet.

Може да се пуска от терминала (python train.py) или да се вика от app.py
чрез train(..., on_epoch_end=callback) за live визуализация.
"""

import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # без прозорец - само запис във файл
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from model import create_model, get_device

# torchvision и sklearn се внасят вътре във функциите: всяка отнема 8-15 с при зареждане
# и не са нужни за Inference. Така приложението стартира по-бързо.

# Статистика на MNIST. Същата нормализация трябва да се ползва и при inference!
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pt"


def get_transform():
    """PIL изображение -> тензор 1x28x28, нормализиран."""
    from torchvision import transforms

    return transforms.Compose([
        transforms.ToTensor(),                       # 0-255 -> 0.0-1.0
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])


def get_dataloaders(batch_size=64, val_size=5000, seed=42):
    """Връща train / val / test DataLoader-и (55k / 5k / 10k)."""
    from torchvision import datasets

    transform = get_transform()
    full_train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    # Фиксиран seed, за да е разделянето еднакво при всеки пуск.
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        full_train, [len(full_train) - val_size, val_size], generator=generator
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=256)
    test_loader = DataLoader(test_set, batch_size=256)
    return train_loader, val_loader, test_loader


def augment_batch(images, max_rotation=15.0, max_shift=0.1, scale_range=(0.9, 1.1)):
    """Случайно въртене, отместване и мащаб на цяла партида (върху GPU).

    Всяка картинка получава своя случайна афинна трансформация. Етикетите не се променят.
    Промените са умерено малки, за да остане цифрата същата (6 не бива да става 9).
    """
    n, device = images.size(0), images.device
    angle = (torch.rand(n, device=device) * 2 - 1) * math.radians(max_rotation)
    scale = torch.empty(n, device=device).uniform_(*scale_range)
    # Координатите на affine_grid са в [-1, 1], т.е. широчината е 2 единици.
    shift = (torch.rand(n, 2, device=device) * 2 - 1) * max_shift * 2

    cos, sin = angle.cos() / scale, angle.sin() / scale
    theta = torch.stack([
        torch.stack([cos, -sin, shift[:, 0]], dim=1),
        torch.stack([sin, cos, shift[:, 1]], dim=1),
    ], dim=1)
    grid = F.affine_grid(theta, list(images.shape), align_corners=False)

    # Работим върху немализирани пиксели (0 = черно), за да е фонът след завъртане черен.
    pixels = images * MNIST_STD + MNIST_MEAN
    out = F.grid_sample(pixels, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return (out - MNIST_MEAN) / MNIST_STD


def train_one_epoch(model, loader, criterion, optimizer, device, should_stop=None, augment=False):
    """Една епоха върху train данните. Връща (loss, accuracy).

    Ако should_stop() стане True, прекъсва рано (върнатите метрики тогава са непълни).
    augment=True прилага augment_batch върху всяка партида (само при обучение).
    """
    model.train()  # Dropout е включен
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        if should_stop and should_stop():
            break
        images, labels = images.to(device), labels.to(device)
        if augment:
            images = augment_batch(images)

        optimizer.zero_grad()              # изчистваме старите градиенти
        outputs = model(images)            # forward pass
        loss = criterion(outputs, labels)  # колко грешим
        loss.backward()                    # backpropagation
        optimizer.step()                   # обновяваме теглата

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    if total == 0:  # спряно преди първата партида
        return 0.0, 0.0
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_predictions=False):
    """Оценка без обучение. Връща (loss, accuracy[, предсказания, етикети])."""
    model.eval()  # Dropout е изключен
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        total_loss += criterion(outputs, labels).item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        if return_predictions:
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    loss, acc = total_loss / total, correct / total
    if return_predictions:
        return loss, acc, torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()
    return loss, acc


def train(epochs=10, batch_size=64, lr=0.001, on_epoch_end=None, save_path=BEST_MODEL_PATH,
          should_stop=None, augment=False, seed=None):
    """Пълно обучение. Връща (model, history).

    on_epoch_end(epoch, history) се вика след всяка епоха - за live графики.
    should_stop() се проверява при всяка партида; ако върне True, недовършената епоха се
    отхвърля и обучението приключва с резултатите от завършените епохи.
    augment=True включва data augmentation върху тренировъчните партиди (не върху validation).
    seed прави началните тегла и разбъркването възпроизводими (за честно сравнение).
    """
    if seed is not None:
        torch.manual_seed(seed)
    device = get_device()
    train_loader, val_loader, _ = get_dataloaders(batch_size)

    model = create_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                                should_stop, augment)
        if should_stop and should_stop():
            break  # недовършена епоха: не я записваме
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Checkpoint: пазим само най-добрия модел по validation accuracy.
        saved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            saved = "  <- запазен"

        print(f"Епоха {epoch:2d}/{epochs} | loss {train_loss:.4f} | acc {train_acc:.4f} | "
              f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | {time.time() - start:.1f}s{saved}")

        if on_epoch_end:
            on_epoch_end(epoch, history)

    return model, history


def test_best_model(save_path=BEST_MODEL_PATH, batch_size=64):
    """Зарежда най-добрия модел и го оценява на test set. Връща (acc, preds, labels)."""
    device = get_device()
    _, _, test_loader = get_dataloaders(batch_size)

    model = create_model(device)
    model.load_state_dict(torch.load(save_path, map_location=device))
    _, acc, preds, labels = evaluate(
        model, test_loader, nn.CrossEntropyLoss(), device, return_predictions=True
    )
    return acc, preds, labels


def save_confusion_matrix(preds, labels, path):
    """Записва confusion matrix като PNG."""
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(7, 7))
    ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return cm


if __name__ == "__main__":
    model, history = train(epochs=10, augment=True)

    (MODELS_DIR / "history.json").write_text(json.dumps(history, indent=2))

    test_acc, preds, labels = test_best_model()
    cm = save_confusion_matrix(preds, labels, MODELS_DIR / "confusion_matrix.png")

    print(f"\nTest accuracy (най-добър модел): {test_acc:.4%}")
    # Най-честите грешки: най-големите стойности извън диагонала.
    errors = [(cm[i, j], i, j) for i in range(10) for j in range(10) if i != j]
    print("Най-чести грешки (истинска -> предсказана):")
    for count, true, pred in sorted(errors, reverse=True)[:5]:
        print(f"  {true} -> {pred}: {count} пъти")
