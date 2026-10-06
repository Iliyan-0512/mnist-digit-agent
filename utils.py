"""Помощни функции: зареждане на модел, обработка на рисунка, предсказване, анализ, графики."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image

from model import create_model, get_device
from train import MNIST_MEAN, MNIST_STD

# Под този контраст (разлика макс-мин, в 0..1) смятаме картинката за празна.
MIN_CONTRAST = 0.2

# Под тази увереност предупреждаваме потребителя. Избрана от измерване: хваща ~половината
# странни входове (шум, драсканици) и отхвърля ~2.5% от реалните цифри. Буквите не хваща.
CONFIDENCE_THRESHOLD = 0.95


def load_model(path, device=None):
    """Зарежда запазените тегла и връща модел в eval режим."""
    device = device or get_device()
    model = create_model(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()  # изключва Dropout
    return model


def _to_grayscale(image):
    """PIL -> 2D float масив 0..1. Прозрачното се слага върху бял фон."""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba)
    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def _thicken_strokes(arr):
    """Удебелява тънките щрихи до дебелината, която има цифра в MNIST.

    В MNIST щрихът е ~2.5px при височина 20px (≈12.5% от размера на цифрата).
    Тънка линия от голяма картинка след смаляване би изчезнала.
    """
    # Работим само върху кутията около цифрата: целият кадър може да е огромен.
    ys, xs = np.where(arr > 0.1)
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    digit_size = max(arr.shape)
    mask = arr > 0.5

    # Дебелина на щриха ~ 2 * площ / брой гранични пиксели.
    inner = mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
    boundary = mask.sum() - inner.sum()
    stroke = 2.0 * mask.sum() / max(boundary, 1)

    extra = 0.125 * digit_size - stroke
    if extra < 1:
        return arr
    kernel = 2 * round(extra / 2) + 1  # нечетен размер на dilation филтъра
    return _max_filter(arr, kernel)


def _max_filter(arr, kernel):
    """Максимум в квадрат kernel x kernel (dilation). Разделен на два прохода: по редове и колони.

    Същото като PIL MaxFilter, но PIL е O(kernel^2) на пиксел и отнема секунди при голямо ядро.
    """
    r = kernel // 2
    # Резултатът е с r пиксела по-голям от всяка страна: щрихът може да се разшири извън кутията.
    padded = np.pad(arr, 2 * r)
    padded = sliding_window_view(padded, kernel, axis=0).max(axis=-1)
    return sliding_window_view(padded, kernel, axis=1).max(axis=-1)


def preprocess_image(image):
    """Превръща произволна рисунка във вида на MNIST.

    Връща (tensor 1x1x28x28, масив 28x28 0..1 за показване)
    или (None, None), ако в картинката няма цифра.
    """
    arr = _to_grayscale(image)

    # Ако ръбовете са светли, фонът е бял -> обръщаме до бяла цифра на черно.
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    if np.median(border) > 0.5:
        arr = 1.0 - arr

    # Разтягаме контраста до 0..1; празна картинка не предсказваме.
    low, high = arr.min(), arr.max()
    if high - low < MIN_CONTRAST:
        return None, None
    arr = (arr - low) / (high - low)

    arr = _thicken_strokes(arr)

    # Изрязваме до кутията около цифрата.
    ys, xs = np.where(arr > 0.1)
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # Мащабираме до най-много 20x20 със запазени пропорции.
    h, w = arr.shape
    scale = 20.0 / max(h, w)
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    small = Image.fromarray((arr * 255).astype(np.uint8)).resize(new_size, Image.LANCZOS)
    small = np.asarray(small, dtype=np.float32) / 255.0
    small = small / small.max()  # смаляването избледнява; връщаме пълната яркост

    # Слагаме в 28x28 и центрираме по център на масата (като в MNIST).
    canvas = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - small.shape[0]) // 2, (28 - small.shape[1]) // 2
    canvas[top:top + small.shape[0], left:left + small.shape[1]] = small

    total = canvas.sum()
    cy = (canvas.sum(axis=1) * np.arange(28)).sum() / total
    cx = (canvas.sum(axis=0) * np.arange(28)).sum() / total
    # В реалния MNIST центърът на масата е на 14.0 (измерено), не 13.5.
    dy, dx = round(14 - cy), round(14 - cx)
    padded = np.pad(canvas, 28)  # нули наоколо, за да няма "прехвърляне" от другата страна
    canvas = padded[28 - dy:56 - dy, 28 - dx:56 - dx]

    tensor = torch.from_numpy((canvas - MNIST_MEAN) / MNIST_STD).float()
    return tensor.unsqueeze(0).unsqueeze(0), canvas


@torch.no_grad()
def predict(model, tensor):
    """Връща масив с 10 вероятности (softmax върху логитите)."""
    device = next(model.parameters()).device
    logits = model(tensor.to(device))
    return torch.softmax(logits, dim=1)[0].cpu().numpy()


@torch.no_grad()
def analyze_model(model):
    """Прекарва целия test set през модела. Връща речник с картинки, предсказания и увереност."""
    from train import get_dataloaders  # вътре: тежко зареждане на torchvision

    device = next(model.parameters()).device
    model.eval()
    _, _, test_loader = get_dataloaders(256)

    images, preds, labels, confs = [], [], [], []
    for x, y in test_loader:
        probs = torch.softmax(model(x.to(device)), dim=1).cpu()
        conf, pred = probs.max(dim=1)
        images.append(x)
        preds.append(pred)
        labels.append(y)
        confs.append(conf)

    return {
        "images": torch.cat(images),
        "preds": torch.cat(preds).numpy(),
        "labels": torch.cat(labels).numpy(),
        "confs": torch.cat(confs).numpy(),
    }


def per_class_report(preds, labels):
    """Таблица с precision, recall и брой примери за всяка цифра."""
    from sklearn.metrics import precision_recall_fscore_support  # вътре: бавно зареждане

    precision, recall, _, support = precision_recall_fscore_support(
        labels, preds, labels=range(10), zero_division=0
    )
    return pd.DataFrame({"цифра": range(10), "precision": precision, "recall": recall, "брой": support})


def plot_metrics(history):
    """Две графики: loss (train/val) и accuracy (train/val) по епохи."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 3.8))

    ax_loss.plot(epochs, history["train_loss"], marker="o", label="train")
    ax_loss.plot(epochs, history["val_loss"], marker="o", label="validation")
    ax_loss.set_title("Loss")

    ax_acc.plot(epochs, [a * 100 for a in history["train_acc"]], marker="o", label="train")
    ax_acc.plot(epochs, [a * 100 for a in history["val_acc"]], marker="o", label="validation")
    ax_acc.set_title("Accuracy (%)")

    for ax in (ax_loss, ax_acc):
        ax.set_xlabel("Епоха")
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_xticks(list(epochs))
    fig.tight_layout()
    return fig


def create_confusion_matrix(preds, labels):
    """Confusion matrix като matplotlib фигура."""
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix  # вътре: бавно зареждане

    fig, ax = plt.subplots(figsize=(6, 6))
    cm = confusion_matrix(labels, preds, labels=range(10))
    ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_xlabel("Предсказана")
    ax.set_ylabel("Истинска")
    fig.tight_layout()
    return fig


def error_gallery(analysis, n=12):
    """Картинки на най-уверените грешки (или None, ако няма грешки)."""
    wrong = np.where(analysis["preds"] != analysis["labels"])[0]
    if len(wrong) == 0:
        return None
    worst = wrong[np.argsort(-analysis["confs"][wrong])][:n]

    cols = 6
    rows = (len(worst) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.3 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, idx in zip(axes.flat, worst):
        pixels = analysis["images"][idx, 0].numpy() * MNIST_STD + MNIST_MEAN  # обратно към 0..1
        ax.imshow(pixels, cmap="gray")
        ax.set_title(f"истина {analysis['labels'][idx]} → {analysis['preds'][idx]}\n"
                     f"{analysis['confs'][idx]:.0%}", fontsize=9)
    fig.tight_layout()
    return fig


def augmentation_preview(indices=(0, 1, 2, 3), per_digit=10):
    """Фигура: за няколко MNIST цифри - оригиналът и случайни варианти от augment_batch."""
    from torchvision import datasets
    from train import DATA_DIR, augment_batch, get_transform

    dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=get_transform())
    fig, axes = plt.subplots(len(indices), per_digit + 1, figsize=(1.0 * (per_digit + 1), 1.05 * len(indices)),
                             squeeze=False)
    for row, idx in zip(axes, indices):
        original = dataset[idx][0].unsqueeze(0)
        variants = augment_batch(original.repeat(per_digit, 1, 1, 1))
        for col, image in enumerate(torch.cat([original, variants])):
            row[col].imshow(image[0].numpy() * MNIST_STD + MNIST_MEAN, cmap="gray", vmin=0, vmax=1)
            row[col].axis("off")
        row[0].set_title("оригинал", fontsize=7)
    fig.tight_layout()
    return fig
