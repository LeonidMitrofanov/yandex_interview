import json
from typing import List, Dict, Any

def create_qa_prompts(input_file: str, output_file: str, prompts_per_chunk: int = 3) -> None:
    """
    Создает унифицированные промты для генерации вопросов из чанков.
    Промты оформлены так, чтобы модель возвращала корректный JSON без лишних пояснений.
    
    Args:
        input_file: путь к файлу с чанками
        output_file: путь для сохранения промтов
        prompts_per_chunk: количество разных промтов на один чанк
    """
    
    with open(input_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    prompts_dataset = []
    
    # Унифицированные шаблоны промтов
    prompt_templates = [
        {
            "name": "factual_questions",
            "template": """Ты — эксперт по книге "Трое из леса" Юрия Никитина. 
На основе приведенного отрывка сгенерируй 2-3 фактологических вопроса, ответы на которые однозначно содержатся в тексте.

Отрывок: {text}

Метаданные: {metadata}

Инструкции:
- Вопросы должны быть конкретными (Кто? Что? Где? Когда?)
- Не задавай интерпретационные или общие вопросы
- Верни только JSON в строго указанном формате

Формат JSON:
{{
  "questions": ["вопрос1", "вопрос2", "вопрос3"]
}}"""
        },
        {
            "name": "reasoning_questions",
            "template": """Ты — внимательный читатель книги "Трое из леса". 
Проанализируй отрывок и создай 2-3 вопроса, проверяющие понимание причинно-следственных связей и мотивов персонажей.

Отрывок: {text}

Метаданные: {metadata}

Инструкции:
- Вопросы должны проверять причины событий, мотивы персонажей (Почему? Зачем?)
- Ответы должны вытекать из содержания отрывка
- Верни только JSON в строго указанном формате

Формат JSON:
{{
  "questions": ["вопрос1", "вопрос2", "вопрос3"]
}}"""
        },
        {
            "name": "detailed_understanding",
            "template": """Ты — специалист по творчеству Юрия Никитина. 
Создай 2-3 глубоких вопроса по отрывку, проверяющих внимательное прочтение и понимание деталей.

Отрывок: {text}

Метаданные: {metadata}

Инструкции:
- Вопросы должны быть уточняющими (Что именно? Каким образом? Какой именно?)
- Проверяй детали, контекст и связи между событиями
- Верни только JSON в строго указанном формате

Формат JSON:
{{
  "questions": ["вопрос1", "вопрос2", "вопрос3"]
}}"""
        }
    ]
    
    for chunk_item in chunks_data:
        chunk = chunk_item["chunk"]
        text = chunk["text"]
        metadata = chunk["metadata"]
        
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
                "source_chunk": text,
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
    
    print("Создаем унифицированные промты...")
    create_qa_prompts(
        input_file=input_file,
        output_file=output_file,
        prompts_per_chunk=3
    )
