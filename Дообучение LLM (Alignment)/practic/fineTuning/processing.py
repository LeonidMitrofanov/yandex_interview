import json
import re
from typing import List, Dict, Any

def split_text_into_chunks(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Разбивает текст на чанки с перекрытием и обрезает по крайним точкам
    
    Args:
        text: исходный текст
        chunk_size: размер чанка в словах
        overlap: перекрытие в словах
    
    Returns:
        Список чанков
    """
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        chunk = ' '.join(words)
        # Обрезаем по точкам для единственного чанка
        chunk = trim_to_sentences(chunk)
        return [chunk] if chunk else []
    
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        
        # Обрезаем чанк по границам предложений
        chunk = trim_to_sentences(chunk)
        
        if chunk:  # Добавляем только непустые чанки
            chunks.append(chunk)
        
        start += chunk_size - overlap
        
        # Если остаток текста меньше chunk_size, но больше overlap
        if start < len(words) and len(words) - start <= chunk_size:
            final_chunk = ' '.join(words[start:])
            final_chunk = trim_to_sentences(final_chunk)
            if final_chunk:
                chunks.append(final_chunk)
            break
    
    return chunks

def trim_to_sentences(text: str) -> str:
    """
    Обрезает текст до ближайших точек слева и справа
    
    Args:
        text: исходный текст
    
    Returns:
        Обрезанный текст или пустая строка если не найдено предложений
    """
    if not text:
        return ""
    
    # Находим первую точку слева
    first_dot = text.find('.')
    if first_dot == -1:
        return ""  # Если нет точек - возвращаем пустую строку
    
    # Находим последнюю точку справа
    last_dot = text.rfind('.')
    if last_dot == -1:
        return ""  # Это не должно случиться, но на всякий случай
    
    # Обрезаем от первой точки до последней точки + 1 (включая саму точку)
    trimmed = text[first_dot + 1:last_dot + 1].strip()
    
    # Убираем возможные начальные точки или запятые после обрезки
    trimmed = re.sub(r'^[.,]\s*', '', trimmed)
    
    return trimmed

def create_chunks_dataset(input_file: str, output_file: str, chunk_size: int = 512, overlap: int = 50) -> None:
    """
    Создает датасет с чанками из исходного JSON
    
    Args:
        input_file: путь к исходному JSON файлу
        output_file: путь для сохранения результата
        chunk_size: размер чанка в словах
        overlap: перекрытие в словах
    """
    
    # Чтение исходного файла
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunks_dataset = []
    
    for part_data in data:
        part_name = part_data.get("part", "Неизвестная часть")
        
        for chapter_data in part_data.get("chapters", []):
            chapter_name = chapter_data.get("chapter", "Неизвестная глава")
            chapter_text = chapter_data.get("text", "")
            
            # Очистка текста от лишних пробелов
            chapter_text = re.sub(r'\s+', ' ', chapter_text).strip()
            
            if not chapter_text:
                continue
            
            # Разбиваем текст главы на чанки
            text_chunks = split_text_into_chunks(chapter_text, chunk_size, overlap)
            
            # Создаем записи для каждого чанка
            for i, chunk_text in enumerate(text_chunks):
                chunk_data = {
                    "chunk": {
                        "text": chunk_text,
                        "metadata": {
                            "author" : "Юрий Никитин",
                            "book_name": "Трое из леса",
                            "part": f"{part_name}",
                            "chapter": f"{chapter_name}"
                        }
                    }
                }
                chunks_dataset.append(chunk_data)
    
    # Сохранение результата
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"Создано {len(chunks_dataset)} чанков")
    print(f"Результат сохранен в: {output_file}")

def analyze_dataset(input_file: str) -> None:
    """
    Анализирует исходный датасет для подбора оптимальных параметров
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_chapters = 0
    total_text_length = 0
    chapter_lengths = []
    
    for part_data in data:
        chapters = part_data.get("chapters", [])
        total_chapters += len(chapters)
        
        for chapter_data in chapters:
            text = chapter_data.get("text", "")
            words = len(text.split())
            chapter_lengths.append(words)
            total_text_length += words
    
    print("=== Анализ датасета ===")
    print(f"Количество частей: {len(data)}")
    print(f"Количество глав: {total_chapters}")
    print(f"Общее количество слов: {total_text_length}")
    print(f"Средняя длина главы: {total_text_length/total_chapters:.0f} слов")
    print(f"Минимальная длина главы: {min(chapter_lengths)} слов")
    print(f"Максимальная длина главы: {max(chapter_lengths)} слов")
    print(f"Медианная длина главы: {sorted(chapter_lengths)[len(chapter_lengths)//2]} слов")

if __name__ == "__main__":
    input_file = "./output/troe_iz_lesa.json"
    output_file = "./output/troe_iz_lesa_chunks.json"
    
    # Сначала анализируем данные
    print("Анализируем исходные данные...")
    analyze_dataset(input_file)
    
    # Создаем чанки с оптимальными параметрами
    print("\nСоздаем чанки...")
    create_chunks_dataset(
        input_file=input_file,
        output_file=output_file,
        chunk_size=500,  # Можно настроить based на анализе
        overlap=50       # 10% перекрытия
    )