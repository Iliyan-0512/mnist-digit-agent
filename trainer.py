"""Обучение във фонова нишка, така че интерфейсът да остава жив.

Модел на нишките:
  - Нишката за обучение е ЕДИНСТВЕНИЯТ, който пише в състоянието (под Lock).
  - Интерфейсът само чете копие чрез snapshot(). Никога не вика Streamlit от нишката.
  - Спирането е заявка (Event), която обучението проверява между партидите.
PyTorch пуска GIL при тежките операции, затова нишка върви реално паралелно със Streamlit.
"""

import copy
import threading

from train import BEST_MODEL_PATH, test_best_model, train

RUNNING, DONE, STOPPED, ERROR = "running", "done", "stopped", "error"


class TrainingJob:
    def __init__(self, epochs, batch_size, lr, save_path, augment=False):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.save_path = save_path
        self.augment = augment

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._state = {
            "status": RUNNING,
            "stopping": False,
            "epoch": 0,
            "history": {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []},
            "test_acc": None,
            "error": None,
        }
        self._thread = threading.Thread(target=self._run, daemon=True, name="training")

    def start(self):
        self._thread.start()

    def stop(self):
        """Заявка за спиране; обучението приключва при следващата партида."""
        with self._lock:
            self._state["stopping"] = True
        self._stop.set()

    def is_running(self):
        return self._thread.is_alive()

    def snapshot(self):
        """Копие на текущото състояние, безопасно за четене от интерфейса."""
        with self._lock:
            return copy.deepcopy(self._state)

    def _on_epoch_end(self, epoch, history):
        with self._lock:
            self._state["epoch"] = epoch
            self._state["history"] = copy.deepcopy(history)

    def _run(self):
        try:
            train(epochs=self.epochs, batch_size=self.batch_size, lr=self.lr,
                  on_epoch_end=self._on_epoch_end, save_path=self.save_path,
                  should_stop=self._stop.is_set, augment=self.augment)

            # Ако поне една епоха е завършила, има запазен модел и можем да го оценим.
            test_acc = None
            if self._state["epoch"] > 0:
                test_acc, _, _ = test_best_model(self.save_path)

            with self._lock:
                self._state["test_acc"] = test_acc
                self._state["status"] = STOPPED if self._stop.is_set() else DONE
        except Exception as exc:  # нишката не бива да умира тихо
            with self._lock:
                self._state["error"] = f"{type(exc).__name__}: {exc}"
                self._state["status"] = ERROR
