import os
import json
import threading
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML
from time import sleep
from random import random

# === НАСТРОЙКИ ===
INPUT_PATH = "output/qa_prompts_detailed.json"
OUTPUT_PATH = "practic/fineTuning/output/qa_results_detailed.json"

FOLDER_ID = "b1ge6b93hbtf0j5b7ptt"
MODEL_NAME = "yandexgpt-lite"
TEMPERATURE = 0.3
MAX_TOKENS = 1024
MAX_WORKERS = 8  # увеличь осторожно если API позволяет
RETRY_ATTEMPTS = 2  # простые повторы для временных ошибок


# === Потокобезопасное сохранение ===
save_lock = threading.Lock()

def save_partial_result(result):
    """Потокобезопасно добавляет результат в JSON-файл (append-стиль)."""
    with save_lock:
        if not os.path.exists(OUTPUT_PATH):
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump([result], f, ensure_ascii=False, indent=2)
            return

        # файл существует
        with open(OUTPUT_PATH, "r+", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []
            except json.JSONDecodeError:
                data = []
            data.append(result)
            f.seek(0)
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.truncate()


# === Парсинг ответа модели ===
def strip_markdown_codeblocks(text: str) -> str:
    """Удаляет ```...``` и `...` блоки, сохраняет содержимое без обёрток."""
    # удалить тройные бэктики с опциональным языком
    text = re.sub(r"```[\s\S]*?```", lambda m: m.group(0).strip("`"), text, flags=re.DOTALL)
    # удалить inline-код `...`
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # убрать лишние ведущие/замыкающие пробелы
    return text.strip()


def extract_first_json_object(text: str):
    """Пытается найти первый {...} фрагмент и распарсить его."""
    # жадно найдём самый большой фрагмент, начинающийся с { и заканчивающийся }
    # подход: найдем все подходящие пары и попробуем распарсить (с длинными в конце)
    matches = list(re.finditer(r"\{[\s\S]*\}", text))
    # попробуем от самых длинных к коротким (чтобы поймать полный JSON)
    matches.sort(key=lambda m: -len(m.group(0)))
    for m in matches:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def extract_questions_array_from_json(obj):
    """Если передали dict, пытаемся достать поле questions."""
    if isinstance(obj, dict):
        if "questions" in obj and isinstance(obj["questions"], list):
            # убедимся, что элементы — строки
            return [str(x).strip() for x in obj["questions"]]
    return None


def try_extract_questions_from_text(text: str):
    """Последовательность стратегий извлечения массива вопросов."""
    original = text
    text = strip_markdown_codeblocks(text)

    # 1) полный JSON-объект
    json_obj = extract_first_json_object(text)
    if json_obj is not None:
        qs = extract_questions_array_from_json(json_obj)
        if qs:
            return qs, "parsed_full_json"

        # если объект не содержит questions, но есть похожие ключи
        for key in ("result", "output", "answers"):
            if key in json_obj and isinstance(json_obj[key], list):
                return [str(x).strip() for x in json_obj[key]], f"parsed_json_key_{key}"

    # 2) найти явный массив "questions": [ ... ]
    m = re.search(r"\"questions\"\s*:\s*(\[[\s\S]*?\])", text)
    if m:
        arr_text = m.group(1)
        try:
            arr = json.loads(arr_text)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr], "parsed_questions_array"
        except Exception:
            # попытка самоочищения: заменить одинарные кавычки на двойные
            try:
                arr_text2 = arr_text.replace("'", "\"")
                arr = json.loads(arr_text2)
                if isinstance(arr, list):
                    return [str(x).strip() for x in arr], "parsed_questions_array_after_replace"
            except Exception:
                pass

    # 3) как крайняя мера — собрать строки, которые выглядят как вопросы (заканчиваются ?)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    question_lines = [ln for ln in lines if ln.endswith("?")]
    if question_lines:
        # уберём нумерацию типа "1. " или "- "
        cleaned = [re.sub(r"^\s*[\-\d\.\)\:]+\s*", "", q).strip() for q in question_lines]
        return cleaned, "extracted_lines_ending_q"

    # 4) попытка найти строки в кавычках, длинные, возможно это ответы
    quoted = re.findall(r"\"([^\"]{10,})\"", text)
    if quoted:
        # отфильтруем короткие/мусор
        filtered = [q.strip() for q in quoted if len(q.strip()) > 10 and q.strip().endswith("?")]
        if filtered:
            return filtered, "extracted_quoted_questions"

    # 5) окончательный fallback — вернём пустой список и статус
    return [], "no_extraction"


# === Обработка одного промта (с ретраями и чисткой) ===
def process_prompt(sdk, item, index):
    """Обрабатывает один промт, парсит ответ, сохраняет результат немедленно."""
    prompt_text = item["prompt"]
    model = None

    attempt = 0
    last_exception = None
    raw_output = ""
    parse_status = ""
    questions = []

    while attempt <= RETRY_ATTEMPTS:
        attempt += 1
        try:
            model = sdk.chat.completions(MODEL_NAME).configure(
                temperature=TEMPERATURE, max_tokens=MAX_TOKENS
            )
            messages = [
                {"role": "system", "text": "Найди ошибки в тексте и исправь их"},
                {"role": "user", "text": prompt_text},
            ]
            result = model.run(messages)
            raw_output = result.text if hasattr(result, "text") else str(result)
            # парсим
            questions, parse_status = try_extract_questions_from_text(raw_output)
            break  # успешно получили ответ (даже если parse failed — мы выйдем и запишем)
        except Exception as e:
            last_exception = e
            # лёгкий экспоненциальный бекоф с джиттером
            sleep_time = (2 ** (attempt - 1)) * 0.5 + random() * 0.3
            sleep(sleep_time)
            continue

    record = {
        "prompt": prompt_text,
        "questions": questions,
        "source_chunk": item.get("source_chunk", ""),
        "metadata": item.get("metadata", {}),
        "prompt_type": item.get("prompt_type", "factual_questions"),
        "raw_output": raw_output,
        "parse_status": parse_status,
        "attempts": attempt,
    }

    if last_exception and not raw_output:
        record["error"] = str(last_exception)

    # Сохраняем результат немедленно
    save_partial_result(record)
    return index


def main():
    load_dotenv()
    api_key = os.getenv("YANDEX_API")

    if not api_key:
        raise ValueError("Ошибка: YANDEX_API не найден в .env файле")

    sdk = YCloudML(folder_id=FOLDER_ID, auth=api_key)
    sdk.setup_default_logging()

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"🔹 Найдено {len(data)} промтов для обработки.\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_prompt, sdk, item, i): i for i, item in enumerate(data, start=1)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Обработка промтов", ncols=100):
            try:
                _ = future.result()
            except Exception as e:
                # на уровне потока — логируем и продолжаем
                print("Ошибка в исполнении потока:", e)

    print(f"\n✅ Все промты обработаны. Результаты (с чисткой) сохранены в {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
