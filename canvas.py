"""Canvas за рисуване на цифра: малък Streamlit компонент (HTML/JS в canvas_component/)."""

import base64
import io
from pathlib import Path

import streamlit.components.v1 as components
from PIL import Image

_component = components.declare_component(
    "digit_canvas", path=str(Path(__file__).parent / "canvas_component")
)


def digit_canvas(key, brush=22):
    """Показва canvas. Връща (PIL картинка или None, номер на рисунката).

    Картинката е бяла цифра на черен фон 280x280. None, ако canvas-ът е празен или изчистен.
    Номерът се променя при всеки нов щрих и при изчистване.
    """
    value = _component(brush=brush, key=key, default=None)
    if not value:
        return None, 0
    number = value.get("n", 0)
    if not value.get("png"):
        return None, number
    data = base64.b64decode(value["png"].split(",", 1)[1])
    return Image.open(io.BytesIO(data)).convert("RGB"), number
