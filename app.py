"""Streamlit интерфейс: обучение с live графики, inference и анализ на модела.

Стартиране:  streamlit run app.py
"""

import shutil
import threading

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image

from agent import AgentError, OllamaBrain, run_agent
from canvas import digit_canvas
from feedback import BACKUP_PATH, FINETUNED_PATH, count_samples, fine_tune, save_sample
from model import count_parameters, create_model, get_device
from train import BEST_MODEL_PATH, MODELS_DIR
from trainer import ERROR, RUNNING, STOPPED, TrainingJob
from utils import (CONFIDENCE_THRESHOLD, analyze_model, augmentation_preview, create_confusion_matrix,
                   error_gallery,
                   load_model, per_class_report, plot_metrics, predict, preprocess_image)

# Обучаваме във временен файл, за да не презапишем добър модел без да искаме.
NEW_MODEL_PATH = MODELS_DIR / "new_model.pt"

st.set_page_config(page_title="MNIST Digit Recognition", page_icon="🎨", layout="wide")


@st.cache_resource
def warm_up():
    """Зарежда тежките библиотеки във фонова нишка, докато потребителят вече вижда страницата.

    torchvision (~14 с) и sklearn (~8 с) иначе се зареждат при първия клик на Training/Analysis.
    Python пази заключване при внасяне, така че ако основната нишка поиска същия модул
    по-рано, просто изчаква, без да се зарежда два пъти.
    """
    def load():
        try:
            import torchvision  # noqa: F401
            import sklearn.metrics  # noqa: F401
        except Exception:
            pass  # ако има проблем, истинската грешка ще се покаже при реалното използване

    threading.Thread(target=load, daemon=True, name="warm-up").start()


@st.cache_resource
def param_count():
    return count_parameters(create_model())


def show_figure(fig):
    st.pyplot(fig)
    plt.close(fig)


def sidebar():
    """Sidebar: режим + информация. Връща избрания режим."""
    st.sidebar.title("🎨 MNIST")
    mode = st.sidebar.radio("Режим", ["Training", "Inference", "Analysis", "Fine-tune", "Agent"])
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Устройство: **{get_device()}**")
    st.sidebar.caption(f"Параметри на модела: **{param_count():,}**")
    job = st.session_state.get("job")
    if job is not None and job.is_running():
        snap = job.snapshot()
        st.sidebar.info(f"⏳ Обучение върви: епоха {snap['epoch']}/{job.epochs}")
    if BEST_MODEL_PATH.exists():
        st.sidebar.success("best_model.pt: наличен")
    else:
        st.sidebar.warning("best_model.pt: още няма")
    return mode


def training_page():
    st.title("Обучение на CNN модела")

    with st.sidebar:
        st.markdown("---")
        st.subheader("Хиперпараметри")
        epochs = st.slider("Епохи", 1, 20, 10)
        batch_size = st.select_slider("Batch size", [16, 32, 64, 128, 256], value=64)
        lr = st.select_slider("Learning rate", [0.0001, 0.0003, 0.001, 0.003, 0.01], value=0.001)
        augment = st.checkbox("Data augmentation", value=True,
                              help="Случайно въртене (±15°), отместване (10%) и мащаб (0.9-1.1) на "
                                   "тренировъчните картинки. Прилага се само при обучение.")

    # Обучението върви в нишка, запазена в session_state: оцелява между rerun-овете
    # и продължава, дори потребителят да отиде на друга страница.
    job = st.session_state.get("job")
    running = job is not None and job.is_running()

    start_col, stop_col, _ = st.columns([1, 1, 4])
    if start_col.button("▶ Start Training", type="primary", disabled=running) and not running:
        job = TrainingJob(epochs, batch_size, lr, NEW_MODEL_PATH, augment)
        st.session_state["job"] = job
        job.start()
        st.rerun()  # презарежда страницата, за да се изключи бутонът Start
    if stop_col.button("⏹ Stop", disabled=not running):
        job.stop()

    with st.expander("Какво прави data augmentation?"):
        if st.button("Покажи примери"):
            show_figure(augmentation_preview())
            st.caption("Първата картинка на всеки ред е оригиналът, останалите са случайни варианти.")

    if job is None:
        st.info("Настрой хиперпараметрите в sidebar и натисни Start Training.")
    elif running:
        live_training_view(job)
    else:
        final_training_view(job)


@st.fragment(run_every="1s")
def live_training_view(job):
    """Само този фрагмент се обновява всяка секунда, а не целият скрипт."""
    snap = job.snapshot()
    if snap["status"] != RUNNING and not job.is_running():
        st.rerun()  # пълен rerun: показва крайния изглед и бутона за запис

    text = "Спирам..." if snap["stopping"] else f"Епоха {snap['epoch']}/{job.epochs}"
    st.progress(snap["epoch"] / job.epochs, text=text)
    show_metrics_line(snap["history"])
    if snap["history"]["train_loss"]:
        show_figure(plot_metrics(snap["history"]))


def show_metrics_line(history):
    if history["train_loss"]:
        st.markdown(
            f"**loss** {history['train_loss'][-1]:.4f} · "
            f"**train acc** {history['train_acc'][-1]:.2%} · "
            f"**val acc** {history['val_acc'][-1]:.2%}"
        )


def final_training_view(job):
    snap = job.snapshot()
    if snap["status"] == ERROR:
        st.error(f"Обучението се провали: {snap['error']}")
        return
    if snap["epoch"] == 0:
        st.warning("Обучението е спряно преди да завърши първата епоха - няма какво да се запази.")
        return

    if snap["status"] == STOPPED:
        st.warning(f"Спряно след {snap['epoch']} от {job.epochs} епохи. Резултатът е от завършените.")
    show_figure(plot_metrics(snap["history"]))
    c1, c2 = st.columns(2)
    c1.metric("Най-добра validation accuracy", f"{max(snap['history']['val_acc']):.2%}")
    c2.metric("Test accuracy", f"{snap['test_acc']:.2%}")

    if st.button("💾 Запази като best_model.pt"):
        shutil.copy(NEW_MODEL_PATH, BEST_MODEL_PATH)
        st.success("Моделът е запазен в models/best_model.pt (презаписан е предишният).")


@st.cache_resource
def get_model(path, mtime):
    """Зарежда модела веднъж. mtime е в ключа, за да се презареди при нов файл."""
    return load_model(path)


def feedback_widget(processed, digit, key):
    """Позволява на потребителя да запази примера с верния етикет за дообучение."""
    st.markdown("**Верен ли е отговорът?** Ако не е, избери истинската цифра и запази примера.")
    col_pick, col_save, _ = st.columns([1, 1, 3])
    true_digit = col_pick.selectbox("Истинска цифра", range(10), index=digit, key=f"label_{key}")
    col_save.write("")  # подравнява бутона с полето
    if col_save.button("💾 Запази пример", key=f"save_{key}"):
        save_sample(processed, true_digit)
        total = sum(count_samples().values())
        st.success(f"Запазен като „{true_digit}“. Общо примери: {total}. Дообучи от режим Fine-tune.")


def show_prediction(image, key="img"):
    """Обработва картинката, предсказва и показва резултата."""
    tensor, processed = preprocess_image(image)
    if tensor is None:
        st.warning("Не виждам цифра в картинката (твърде празна или без контраст).")
        return

    model = get_model(str(BEST_MODEL_PATH), BEST_MODEL_PATH.stat().st_mtime)
    probs = predict(model, tensor)
    digit = int(probs.argmax())

    col_in, col_proc, col_out = st.columns([1, 1, 2])
    col_in.image(image, caption="Оригинал", width=160)
    col_proc.image(processed, caption="Какво вижда моделът (28×28)", width=160, clamp=True)
    with col_out:
        st.metric("Предсказание", digit, f"{probs[digit]:.1%} увереност", delta_color="off")
        st.bar_chart(pd.DataFrame({"вероятност": probs}, index=[str(i) for i in range(10)]))

    if probs[digit] < CONFIDENCE_THRESHOLD:
        st.warning(f"Ниска увереност (под {CONFIDENCE_THRESHOLD:.0%}): рисунката може да е нечетлива "
                   "или да не е цифра. Моделът винаги избира някоя цифра, дори да му покажеш нещо друго.")

    feedback_widget(processed, digit, key)

def inference_page():
    st.title("Разпознаване на цифри")

    if not BEST_MODEL_PATH.exists():
        st.error("Няма обучен модел. Обучи го от режим Training.")
        return

    draw_tab, upload_tab = st.tabs(["✏️ Рисувай", "📁 Качи файл"])
    with draw_tab:
        draw_inference()
    with upload_tab:
        upload_inference()


def draw_inference():
    st.caption("Нарисувай една цифра с мишката или пръста. Предсказанието се обновява след всеки щрих. "
               "Рисувай голямо и с плътни щрихи.")
    image, number = digit_canvas(key="digit_canvas")
    if image is None:
        st.info("Нарисувай цифра в черното поле.")
        return
    show_prediction(image, key=f"draw_{number}")


def upload_inference():
    st.caption("Качи картинка с една цифра (тъмна на светъл фон или светла на тъмен).")
    uploaded = st.file_uploader("Изображение", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        return
    try:
        image = Image.open(uploaded)
        image.load()  # принуждава декодиране сега, за да хванем повредени файлове
    except Exception:
        st.error("Файлът не може да се прочете като изображение.")
        return
    show_prediction(image, key=uploaded.file_id)


@st.cache_resource
def get_analysis(path, mtime):
    """Оценява модела върху целия test set (веднъж за версия на файла)."""
    return analyze_model(get_model(path, mtime))


def analysis_page():
    st.title("Анализ на модела")

    if not BEST_MODEL_PATH.exists():
        st.error("Няма обучен модел. Обучи го от режим Training.")
        return

    with st.spinner("Оценявам върху test set (10 000 картинки)..."):
        a = get_analysis(str(BEST_MODEL_PATH), BEST_MODEL_PATH.stat().st_mtime)

    errors = int((a["preds"] != a["labels"]).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Test accuracy", f"{1 - errors / len(a['labels']):.2%}")
    c2.metric("Грешни от 10 000", errors)
    c3.metric("Средна увереност", f"{a['confs'].mean():.1%}")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Confusion matrix")
        fig = create_confusion_matrix(a["preds"], a["labels"])
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("По цифри")
        st.caption("Recall: колко от истинските X намери. Precision: колко от наречените X са наистина X.")
        report = per_class_report(a["preds"], a["labels"])
        st.dataframe(report.style.format({"precision": "{:.2%}", "recall": "{:.2%}"}),
                     hide_index=True, width="stretch")

    st.subheader("Най-уверените грешки")
    st.caption("Формат: истинска → предсказана, и колко е бил сигурен моделът.")
    fig = error_gallery(a)
    if fig is None:
        st.success("Няма грешки.")
    else:
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Граници на модела")
    st.info(
        "Моделът има само 10 изхода и **винаги избира цифра**. При проверка със синтетични входове "
        "(≈300 букви, драсканици и шум) предупреждението за ниска увереност хвана около половината; "
        "**буквите често се разпознават като цифри с над 99% увереност** (напр. I→1, O→0). "
        "Висока увереност не е гаранция за верен отговор."
    )


def pct(value):
    return "-" if value is None else f"{value:.2%}"


def finetune_page():
    st.title("Дообучение с твои примери")
    st.caption("Запазвай примери от режим Inference (с верния етикет), после дообучи модела върху тях. "
               "Новият модел се пази отделно и ти решаваш дали да го приемеш.")

    counts = count_samples()
    total = sum(counts.values())
    if total == 0:
        st.info("Още няма запазени примери. Отиди в Inference, качи картинка и натисни „Запази пример“.")
        return

    st.metric("Запазени примери", total)
    st.bar_chart(pd.DataFrame({"брой": counts}).rename_axis("цифра"))
    present = [v for v in counts.values() if v]
    if total >= 10 and max(present) > 3 * min(present):
        st.warning("Примерите са силно небалансирани между цифрите. Моделът може да се наклони към "
                   "цифрите, на които има най-много.")

    if st.button("🔧 Дообучи", type="primary"):
        with st.spinner("Дообучавам (няколко секунди)..."):
            st.session_state["finetune"] = fine_tune()

    result = st.session_state.get("finetune")
    if not result:
        return

    b, a = result["before"], result["after"]
    st.subheader("Резултат")
    st.caption(f"{result['n_train']} примера за учене (повторени ×{result['repeat']}) + "
               f"{result['replay']} стари MNIST картинки. Държани настрана: {result['n_holdout']}.")
    rows = [("MNIST test (регресия: не бива да пада)", b["mnist_test"], a["mnist_test"]),
            ("Твоите примери (учени, не доказва обобщаване)", b["feedback_train"], a["feedback_train"])]
    if result["n_holdout"]:
        rows.append(("Държани настрана (честна оценка)", b["feedback_holdout"], a["feedback_holdout"]))
    else:
        st.info("Под 20 примера няма държани настрана, затова няма честна оценка на подобрението.")
    st.table(pd.DataFrame(rows, columns=["Метрика", "Преди", "След"]).assign(
        Преди=lambda d: d["Преди"].map(pct), След=lambda d: d["След"].map(pct)).set_index("Метрика"))

    if a["mnist_test"] < b["mnist_test"] - 0.003:
        st.warning("Точността на MNIST падна с повече от 0,3 пункта: моделът е започнал да забравя.")

    if st.button("✅ Приеми новия модел"):
        shutil.copy(BEST_MODEL_PATH, BACKUP_PATH)
        shutil.copy(FINETUNED_PATH, BEST_MODEL_PATH)
        st.session_state.pop("finetune", None)
        st.success("Приет. Предишният модел е запазен като models/best_model.backup.pt.")


def show_agent_event(container, event):
    """Показва едно събитие на агента (извикване на инструмент, резултат или краен отговор)."""
    if event["type"] == "call":
        arguments = ", ".join(f"{k}={v!r}" for k, v in event["arguments"].items())             if isinstance(event["arguments"], dict) else str(event["arguments"])
        container.markdown(f"🔧 **{event['name']}**({arguments})")
    elif event["type"] == "result":
        container.code(event["result"], language="json")


def agent_page():
    st.title("AI агент")
    st.caption("Дай цел на естествен език. Езиков модел решава сам кои инструменти да извика и в какъв ред, "
               "вижда резултатите и решава следващата стъпка, докато стигне до отговор. "
               "Картинките са в папката agent_files/.")

    with st.sidebar:
        st.markdown("---")
        st.subheader("Агент")
        model_name = st.text_input("Ollama модел", value="qwen2.5:3b")
        host = st.text_input("Ollama адрес", value="http://localhost:11434")

    brain = OllamaBrain(model_name, host)
    try:
        installed = brain.available_models()
    except AgentError as exc:
        st.error(f"{exc}\n\nСтартирай в отделен терминал: `ollama serve`")
        return
    if model_name not in installed and f"{model_name}:latest" not in installed:
        st.warning(f"Моделът „{model_name}“ не е изтеглен. Изтегли го с `ollama pull {model_name}`. "
                   f"Налични: {', '.join(installed) or 'няма'}.")
        return

    st.caption("Примери: „Разпознай всички картинки и кажи кои са съмнителни.“ · "
               "„Колко примера за дообучение имам?“ · „Колко е точността на модела?“")
    goal = st.chat_input("Дай цел на агента...")

    if goal:
        with st.chat_message("user"):
            st.write(goal)
        with st.chat_message("assistant"):
            status = st.status("Агентът работи...", expanded=True)
            events = []

            def on_event(event):
                events.append(event)
                show_agent_event(status, event)

            try:
                outcome = run_agent(goal, brain, on_event=on_event)
            except AgentError as exc:
                status.update(label="Грешка", state="error")
                st.error(str(exc))
                return
            status.update(label=f"Готово след {outcome['steps']} стъпки", state="complete", expanded=False)
            st.write(outcome["final"])
        st.session_state["agent_run"] = {"goal": goal, "events": events, "outcome": outcome}
        return

    saved = st.session_state.get("agent_run")
    if saved:  # показваме последния резултат при следващ rerun
        with st.chat_message("user"):
            st.write(saved["goal"])
        with st.chat_message("assistant"):
            with st.status(f"Готово след {saved['outcome']['steps']} стъпки", state="complete", expanded=False) as s:
                for event in saved["events"]:
                    show_agent_event(s, event)
            st.write(saved["outcome"]["final"])


warm_up()
mode = sidebar()
if mode == "Training":
    training_page()
elif mode == "Inference":
    inference_page()
elif mode == "Analysis":
    analysis_page()
elif mode == "Fine-tune":
    finetune_page()
else:
    agent_page()
