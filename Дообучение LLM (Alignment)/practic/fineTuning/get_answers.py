import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML
from time import sleep
from random import random

# === НАСТРОЙКИ ===
INPUT_PATH = "output/dataset.jsonl"
OUTPUT_PATH = "output/output_with_model_responses.jsonl"
FOLDER_ID = "b1ge6b93hbtf0j5b7ptt"  # Замени на твой folder_id
MODEL_NAME = "yandexgpt-lite"
TEMPERATURE = 0.3
MAX_TOKENS = 1024
MAX_WORKERS = 8
RETRY_ATTEMPTS = 2

# === Потокобезопасное сохранение ===
save_lock = threading.Lock()

def save_partial_result(result):
    """Потокобезопасно добавляет результат в JSONL файл."""
    with save_lock:
        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

def create_context_aware_prompt(user_question, context_text):
    system_prompt = """Ты - помощник, который отвечает на вопросы ИСКЛЮЧИТЕЛЬНО на основе предоставленного текста. 

ПРАВИЛА:
1. Отвечай ТОЛЬКО на основе информации из предоставленного текста
2. Если в тексте нет информации для ответа на вопрос, скажи "В предоставленном тексте нет информации об этом"
3. Не используй свои знания вне контекста
4. Не придумывай информацию
5. Будь точным и лаконичным

ТЕКСТ ДЛЯ ОТВЕТА:
"""
    
    user_prompt = f"""ВОПРОС: {user_question}

ОТВЕТ (на основе только предоставленного текста):"""
    
    return [
        {"role": "system", "text": system_prompt + context_text},
        {"role": "user", "text": user_prompt}
    ]

def extract_context_from_item(item):
    context_parts = []
    for req in item.get("request", []):
        if "text" in req:
            context_parts.append(req["text"])
    if "response" in item:
        context_parts.append(item["response"])
    return "\n\n".join(context_parts)

def process_item(sdk, item, index):
    user_question = item["request"][0]["text"]
    context_text = extract_context_from_item(item)
    
    model_response = ""
    last_exception = None
    attempt = 0

    while attempt <= RETRY_ATTEMPTS:
        attempt += 1
        try:
            model = sdk.chat.completions(MODEL_NAME).configure(
                temperature=TEMPERATURE, max_tokens=MAX_TOKENS
            )
            messages = create_context_aware_prompt(user_question, context_text)
            result = model.run(messages)
            model_response = result.text if hasattr(result, "text") else str(result)
            break
        except Exception as e:
            last_exception = e
            sleep_time = (2 ** (attempt - 1)) * 0.5 + random() * 0.3
            sleep(sleep_time)
            continue

    # Сохраняем только request и response
    new_item = {
        "request": item["request"],
        "response": model_response if model_response else item.get("response", "")
    }

    if last_exception and not model_response:
        new_item["response"] = item.get("response", "")

    save_partial_result(new_item)
    return index

def main():
    load_dotenv()
    api_key = os.getenv("YANDEX_API")
    
    if not api_key:
        raise ValueError("Ошибка: YANDEX_API не найден в .env файле")
    
    sdk = YCloudML(folder_id=FOLDER_ID, auth=api_key)
    sdk.setup_default_logging()
    
    # Читаем JSONL файл
    data = []
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"🔹 Найдено {len(data)} элементов для обработки.\n")
    
    # Очищаем выходной файл если он существует
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_item, sdk, item, i): i for i, item in enumerate(data, start=1)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Обработка элементов", ncols=100):
            try:
                _ = future.result()
            except Exception as e:
                print("Ошибка в потоке:", e)
    
    print(f"\n✅ Все элементы обработаны. Результаты сохранены в {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
