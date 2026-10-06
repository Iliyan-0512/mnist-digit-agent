"""CNN архитектура за разпознаване на ръкописни цифри (MNIST)."""

import torch
import torch.nn as nn


class MNISTNet(nn.Module):
    """Малка конволюционна мрежа: 2 conv блока + 2 линейни слоя.

    Вход:  (batch, 1, 28, 28)  - сиви изображения, нормализирани
    Изход: (batch, 10)         - логити за цифрите 0-9
    """

    def __init__(self):
        super().__init__()

        # Конволюционна част: извлича шарки (линии, ъгли, извивки).
        # padding=1 запазва размера след 3x3 филтър (28 -> 28).
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 1x28x28  -> 32x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 32x28x28 -> 32x14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32x14x14 -> 64x14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 64x14x14 -> 64x7x7
        )

        # Класификатор: решава коя цифра е, на база намерените шарки.
        self.classifier = nn.Sequential(
            nn.Flatten(),                                 # 64x7x7 -> 3136
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),                              # изключено при model.eval()
            nn.Linear(128, 10),                           # логити, без softmax
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_device():
    """GPU (CUDA), ако има, иначе CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model(device=None):
    """Създава нов модел и го слага на подходящото устройство."""
    device = device or get_device()
    return MNISTNet().to(device)


def count_parameters(model):
    """Брой на обучаемите параметри."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Бърз тест: подаваме фалшива партида и гледаме формите.
    model = create_model()
    device = next(model.parameters()).device
    x = torch.randn(4, 1, 28, 28, device=device)
    out = model(x)
    print("Device:    ", device)
    print("Вход:      ", tuple(x.shape))
    print("Изход:     ", tuple(out.shape))
    print("Параметри: ", f"{count_parameters(model):,}")
    assert out.shape == (4, 10)
    print("OK")
