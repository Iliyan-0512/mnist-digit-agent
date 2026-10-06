"""Инструменти на агента: обикновени функции върху кода на проекта.

Всяка функция връща текст (JSON), който се връща на модела като резултат.
Агентът е ограничен до папката agent_files/ и НЯМА инструмент за замяна на best_model.pt:
решението дали да се приеме дообучен модел остава за човека.
"""

import json
from functools import lru_cache

from PIL import Image

from feedback import count_samples, fine_tune, save_sample
from train import BASE_DIR, BEST_MODEL_PATH
from utils import CONFIDENCE_THRESHOLD, analyze_model, load_model, predict, preprocess_image

AGENT_DIR = BASE_DIR / "agent_files"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
MAX_RESULT_CHARS = 3000


def _safe_path(filename):
    """Път до файл вътре в agent_files/. Отхвърля излизане от папката (../) и липсващи файлове."""
    root = AGENT_DIR.resolve()
    path = (root / str(filename)).resolve()
    if root not in path.parents:
        raise ValueError("Достъпът е позволен само до файлове в папката agent_files.")
    if not path.is_file():
        raise FileNotFoundError(f"Няма такъв файл: {filename}")
    return path


@lru_cache(maxsize=1)
def _model_for(mtime):
    return load_model(BEST_MODEL_PATH)


def _model():
    return _model_for(BEST_MODEL_PATH.stat().st_mtime)  # нов файл -> презарежда


def _load_processed(filename):
    image = Image.open(_safe_path(filename))
    image.load()
    return preprocess_image(image)


# --- инструментите ---------------------------------------------------------------

def list_images():
    """Имената на картинките в папката agent_files."""
    AGENT_DIR.mkdir(exist_ok=True)
    names = sorted(p.name for p in AGENT_DIR.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return json.dumps({"images": names, "count": len(names)}, ensure_ascii=False)


def classify_image(filename):
    """Разпознава цифра. Връща цифра, увереност, 3-те най-вероятни и флаг дали е съмнително."""
    tensor, _ = _load_processed(filename)
    if tensor is None:
        return json.dumps({"filename": filename, "error": "В картинката не се вижда цифра."},
                          ensure_ascii=False)
    probs = predict(_model(), tensor)
    top = probs.argsort()[::-1][:3]
    return json.dumps({
        "filename": filename,
        "digit": int(top[0]),
        "confidence": round(float(probs[top[0]]), 3),
        "top3": [{"digit": int(d), "probability": round(float(probs[d]), 3)} for d in top],
        "suspicious": bool(probs[top[0]] < CONFIDENCE_THRESHOLD),
    }, ensure_ascii=False)


def classify_all_images():
    """Разпознава всички картинки в папката наведнъж и връща готов списък на съмнителните."""
    AGENT_DIR.mkdir(exist_ok=True)
    names = sorted(p.name for p in AGENT_DIR.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    results = []
    for name in names:
        tensor, _ = _load_processed(name)
        if tensor is None:
            results.append({"filename": name, "digit": None, "confidence": None, "suspicious": True,
                            "note": "не се вижда цифра"})
            continue
        probs = predict(_model(), tensor)
        results.append({"filename": name, "digit": int(probs.argmax()),
                        "confidence": round(float(probs.max()), 3),
                        "suspicious": bool(probs.max() < CONFIDENCE_THRESHOLD)})
    return json.dumps({
        "total": len(results),
        "suspicious_files": [r["filename"] for r in results if r["suspicious"]],
        "results": results,
    }, ensure_ascii=False)


def save_example(filename, label):
    """Запазва картинката като пример за дообучение с верния етикет (0-9)."""
    label = int(label)
    if not 0 <= label <= 9:
        raise ValueError("label трябва да е цяло число от 0 до 9.")
    tensor, processed = _load_processed(filename)
    if tensor is None:
        raise ValueError("В картинката не се вижда цифра, не може да се запази.")
    save_sample(processed, label)
    return json.dumps({"saved": filename, "label": label, "total_examples": sum(count_samples().values())},
                      ensure_ascii=False)


def feedback_status():
    """Колко примера за дообучение има запазени по цифри."""
    counts = count_samples()
    return json.dumps({"per_digit": counts, "total": sum(counts.values())}, ensure_ascii=False)


def evaluate_model():
    """Оценява текущия best_model.pt върху 10 000 тестови картинки."""
    result = analyze_model(_model())
    errors = int((result["preds"] != result["labels"]).sum())
    total = len(result["labels"])
    return json.dumps({"test_accuracy": round(1 - errors / total, 4), "errors": errors, "total": total})


def fine_tune_model():
    """Дообучава модела върху запазените примери. НЕ замества best_model.pt (това решава човек)."""
    r = fine_tune()
    b, a = r["before"], r["after"]
    return json.dumps({
        "examples_used": r["n_samples"],
        "mnist_test_accuracy": {"before": round(b["mnist_test"], 4), "after": round(a["mnist_test"], 4)},
        "own_examples_accuracy": {"before": round(b["feedback_train"], 3), "after": round(a["feedback_train"], 3)},
        "holdout_accuracy": None if not r["n_holdout"] else
            {"before": round(b["feedback_holdout"], 3), "after": round(a["feedback_holdout"], 3)},
        "note": "Новият модел е записан отделно и НЕ е приет. Човек решава дали да го приеме.",
    }, ensure_ascii=False)


# --- регистър: функция + описание, което моделът чете, за да избере инструмент -------------

TOOLS = {
    "list_images": (list_images, "List the image files available in the agent_files folder.", {}),
    "classify_all_images": (classify_all_images,
        "Recognize the digits in ALL images of the agent_files folder in one call. Returns every "
        "file's digit and confidence plus 'suspicious_files', the list of files with low confidence. "
        "Use this when asked about all images or the whole folder.", {}),
    "classify_image": (classify_image,
        "Recognize the handwritten digit in ONE specific image file. Returns the digit, a confidence "
        "from 0 to 1, the top 3 candidates and a 'suspicious' flag set when confidence is low.",
        {"filename": ("string", "Image file name from list_images")}),
    "save_example": (save_example,
        "Save an image as a labeled training example for later fine-tuning. Only call this "
        "when the user has told you the correct digit.",
        {"filename": ("string", "Image file name"), "label": ("integer", "The correct digit, 0-9")}),
    "feedback_status": (feedback_status,
        "Show how many labeled examples are saved for each digit.", {}),
    "evaluate_model": (evaluate_model,
        "Measure the accuracy of the current model on the 10000-image test set.", {}),
    "fine_tune_model": (fine_tune_model,
        "Fine-tune the model on the saved examples and report accuracy before and after. "
        "Does NOT replace the current model; a human decides that.", {}),
}


def tool_schemas():
    """Описанията на инструментите във формата, който очакват Ollama/OpenAI-подобните API-та."""
    schemas = []
    for name, (_, description, params) in TOOLS.items():
        schemas.append({"type": "function", "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {p: {"type": t, "description": d} for p, (t, d) in params.items()},
                "required": list(params),
            },
        }})
    return schemas


def execute_tool(name, arguments):
    """Изпълнява инструмент безопасно. Грешките се връщат като текст на модела, не го сриват."""
    if name not in TOOLS:
        return f"ERROR: непознат инструмент '{name}'. Налични: {', '.join(TOOLS)}."
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return "ERROR: аргументите не са валиден JSON."
    if not isinstance(arguments, dict):
        return "ERROR: аргументите трябва да са обект."

    function, _, params = TOOLS[name]
    missing = [p for p in params if p not in arguments]
    unknown = [a for a in arguments if a not in params]
    if missing or unknown:
        return f"ERROR: грешни аргументи (липсват: {missing}, непознати: {unknown}). Очаквани: {list(params)}."
    try:
        result = function(**arguments)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"
    return result if len(result) <= MAX_RESULT_CHARS else result[:MAX_RESULT_CHARS] + "...(съкратено)"
