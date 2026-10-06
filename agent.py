"""Агентът: цикъл "мозък решава -> инструмент действа -> мозъкът вижда резултата".

Мозъкът е сменяем: всичко, което има метод chat(messages, tools), става мозък.
Тук е OllamaBrain (локален модел). Цикълът не знае нищо за Streamlit.
"""

import requests

from agent_tools import execute_tool, tool_schemas

MAX_STEPS = 15

SYSTEM_PROMPT = """You are an assistant that manages a handwritten digit recognition system.
You can only act through the provided tools. Rules:
- ACT, do not announce. Never write plans or promises such as "now I will...". If more tool
  calls are needed to finish the task, make them immediately, one after another.
- Give the final answer only after you have called every tool needed and hold all the data.
- Use tools to find out facts. Never invent file names, digits or numbers; report only what tool results say.
- To check all images, call classify_all_images (a single call) and report its "suspicious_files" list.
  To check one specific file, call classify_image with that file name. Use list_images only to see file names.
- Call save_example only when the user has told you the correct digit for that image.
- You cannot replace the production model. After fine_tune_model, report the before/after numbers
  and tell the user that a human must decide whether to accept the new model.
- If a tool returns an ERROR, read it and correct your next call instead of repeating it.
Always write your final answer in Bulgarian."""


class AgentError(Exception):
    """Проблем с мозъка (напр. Ollama не отговаря), а не с инструмент."""


class OllamaBrain:
    """Мозък през локален Ollama сървър (HTTP API)."""

    def __init__(self, model="qwen2.5:3b", host="http://localhost:11434", timeout=600):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout

    def available_models(self):
        """Имената на изтеглените модели; AgentError, ако сървърът не отговаря."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise AgentError(f"Ollama не отговаря на {self.host}. Стартиран ли е `ollama serve`? ({exc})")
        return [m["name"] for m in response.json().get("models", [])]

    def chat(self, messages, tools):
        """Една стъпка на мозъка. Връща съобщението на асистента (може да съдържа tool_calls)."""
        try:
            response = requests.post(
                f"{self.host}/api/chat", timeout=self.timeout,
                json={"model": self.model, "messages": messages, "tools": tools,
                      "stream": False, "options": {"temperature": 0}},
            )
        except requests.RequestException as exc:
            raise AgentError(f"Ollama не отговаря на {self.host}: {exc}")
        if response.status_code != 200:
            raise AgentError(f"Ollama върна грешка {response.status_code}: {response.text[:300]}")
        return response.json()["message"]


def run_agent(goal, brain, max_steps=MAX_STEPS, on_event=None):
    """Изпълнява цел. Връща речник: final (текст), steps (брой), stopped (защо е спрял).

    on_event(event) се вика за всяко събитие, за да може интерфейсът да го показва на живо:
      {"type": "call", "name", "arguments"} | {"type": "result", "name", "result"}
      | {"type": "final", "text"}
    """
    emit = on_event or (lambda event: None)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": goal}]
    tools = tool_schemas()

    for step in range(1, max_steps + 1):
        message = brain.chat(messages, tools)
        messages.append(message)

        calls = message.get("tool_calls") or []
        if not calls:  # мозъкът не иска инструмент -> това е крайният отговор
            text = (message.get("content") or "").strip()
            emit({"type": "final", "text": text})
            return {"final": text, "steps": step, "stopped": "answered"}

        for call in calls:
            function = call.get("function", {})
            name, arguments = function.get("name", ""), function.get("arguments", {})
            emit({"type": "call", "name": name, "arguments": arguments})
            result = execute_tool(name, arguments)
            emit({"type": "result", "name": name, "result": result})
            messages.append({"role": "tool", "tool_name": name, "content": result})

    text = f"Спрях след {max_steps} стъпки, без да стигна до краен отговор (ограничение срещу безкраен цикъл)."
    emit({"type": "final", "text": text})
    return {"final": text, "steps": max_steps, "stopped": "max_steps"}
