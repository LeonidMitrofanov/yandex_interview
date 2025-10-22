import json
from typing import List, Dict, Any

def create_qa_prompts(input_file: str, output_file: str, prompts_per_chunk: int = 3) -> None:
    """
    Создает промты для генерации вопросов из чанков
    
    Args:
        input_file: путь к файлу с чанками
        output_file: путь для сохранения промтов
        prompts_per_chunk: количество разных промтов на один чанк
    """
    
    with open(input_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    prompts_dataset = []
    
    # Разные типы промтов для разнообразия вопросов
    prompt_templates = [
        {
            "name": "factual_questions",
            "template": """Ты — эксперт по книге "Трое из леса" Юрия Никитина. На основе приведенного отрывка сгенерируй 2-3 фактологических вопроса, ответы на которые непосредственно содержатся в тексте.

Отрывок: {text}

Метаданные: {metadata}

Инструкции:
- Вопросы должны быть конкретными и фактологическими (Кто? Что? Где? Когда?)
- Ответ должен однозначно содержаться в отрывке
- Избегай общих и интерпретационных вопросов
- Формулируй вопросы так, чтобы они были самодостаточными

Сгенерируй вопросы в формате JSON:
{{
  "questions": ["вопрос1", "вопрос2", "вопрос3"]
}}"""
        },
        {
            "name": "reasoning_questions", 
            "template": """Ты — внимательный читатель книги Юрия Никитина "Трое из леса". Проанализируй отрывок и создай 2-3 вопроса, требующие понимания причинно-следственных связей и мотивов персонажей.

Текст: {text}

Контекст: {metadata}

Требования к вопросам:
- Фокусируйся на причинах событий и мотивах персонажей (Почему? Зачем? С какой целью?)
- Вопросы должны проверять понимание логики повествования
- Ответы должны вытекать из содержания отрывка

Верни результат в JSON:
{{
  "questions": ["вопрос1", "вопрос2"] 
}}"""
        },
        {
            "name": "detailed_understanding",
            "template": """Как специалист по творчеству Юрия Никитина, создай глубокие вопросы по приведенному отрывку из "Трое из леса", которые проверяют внимательное прочтение и понимание деталей.

Отрывок из книги: {text}

{metadata}

Создай 2-3 вопроса следующих типов:
- Уточняющие (Каким образом? Что именно? Какой именно?)
- На понимание деталей и контекста
- На установление связей между событиями

Каждый вопрос должен иметь четкий ответ в тексте.

Формат вывода:
{{
  "questions": ["вопрос1", "вопрос2", "вопрос3"]
}}"""
        }
    ]
    
    for chunk_item in chunks_data:
        chunk = chunk_item["chunk"]
        text = chunk["text"]
        metadata = chunk["metadata"]
        
        # Создаем несколько промтов для каждого чанка (для разнообразия)
        for i in range(min(prompts_per_chunk, len(prompt_templates))):
            template = prompt_templates[i]
            prompt_text = template["template"].format(
                text=text,
                metadata=metadata
            )
            
            prompt_data = {
                "prompt": prompt_text,
                "metadata": metadata,
                "prompt_type": template["name"],
                "source_chunk": text,  # Сохраняем часть исходный chunk
                "expected_format": "json"
            }
            
            prompts_dataset.append(prompt_data)
    
    # Сохраняем результат
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(prompts_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"Создано {len(prompts_dataset)} промтов")
    print(f"Типы промтов: {[t['name'] for t in prompt_templates[:prompts_per_chunk]]}")



if __name__ == "__main__":
    input_file = "output/troe_iz_lesa_chunks.json"
    output_file = "output/qa_prompts_detailed.json"
    
    # Вариант 1: Разнообразные промты (рекомендуется)
    print("Создаем разнообразные промты...")
    create_qa_prompts(
        input_file=input_file,
        output_file=output_file,
        prompts_per_chunk=3
    )