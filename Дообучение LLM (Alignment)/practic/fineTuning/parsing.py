import json
import re
from bs4 import BeautifulSoup

def extract_text_with_parts(file_path):
    with open(file_path, 'r', encoding='windows-1251') as file:
        content = file.read()
    
    soup = BeautifulSoup(content, 'html.parser')
    for tag in soup.find_all(['meta', 'a', 'script', 'style']):
        tag.decompose()
    
    # сохраняем переносы строк между блоками
    full_text = soup.get_text(separator='\n')
    
    # удаляем остаточные "<...>" если они попали в текст и нормализуем CRLF
    full_text = re.sub(r'</?[^>]+>', '', full_text)
    full_text = full_text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Убираем лишние пробелы внутри строк, но НЕ трогаем разделение на строки
    lines = [re.sub(r'\s+', ' ', line).strip() for line in full_text.split('\n')]
    # Оставляем только непустые строки (но сохраняем структуру)
    lines = [line for line in lines if line]
    
    parts = []
    current_part = None
    current_chapter = None
    current_content = []
    
    for text in lines:
        # ЧАСТЬ
        if re.match(r'^\* ЧАСТЬ [А-Я]+ \*$', text):
            if current_part and current_part["chapters"]:
                parts.append(current_part)
            current_part = {"part": text.replace('*', '').strip(), "chapters": []}
            current_chapter = None
            current_content = []
            continue

        # ГЛАВА (подходит "Глава 1" и "Глава 12")
        if re.match(r'^Глава\s+\d+\.?$', text, flags=re.IGNORECASE):
            if current_chapter and current_content and current_part is not None:
                chapter_text = '\n'.join(current_content).strip()
                if chapter_text:
                    current_part["chapters"].append({
                        "chapter": current_chapter,
                        "text": chapter_text
                    })
            current_chapter = text
            current_content = []
            continue

        # ТЕКСТ ГЛАВЫ
        if current_part and current_chapter:
            if not any(meta in text for meta in [
                'Юрий Никитин', 'Copyright', 'http://',
                'Email:', 'Оригинал', 'Трое из леса'
            ]):
                current_content.append(text)

    # сохраняем последнюю главу и часть
    if current_chapter and current_content and current_part is not None:
        chapter_text = '\n'.join(current_content).strip()
        if chapter_text:
            current_part["chapters"].append({
                "chapter": current_chapter,
                "text": chapter_text
            })
    if current_part and current_part["chapters"]:
        parts.append(current_part)

    return parts

def save_to_json(parts, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(parts, file, ensure_ascii=False, indent=2)

def print_json_structure(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    print(f"Всего частей: {len(data)}")
    total_chapters = 0

    # Выводим структуру для проверки
    for i, part in enumerate(data, 1):
        print(f"\nЧасть {i}: {part['part']}")
        for j, chapter in enumerate(part["chapters"], 1):
            print(f"  Глава {j}: {chapter['chapter']}")
        total_chapters += len(part["chapters"])

    print(f"\nОбщее количество глав: {total_chapters}")

# Основная часть скрипта
if __name__ == "__main__":
    input_file = "input/Troe_iz_lesa.htm"
    output_file = "output/troe_iz_lesa.json"
    
    try:
        parts = extract_text_with_parts(input_file)
        save_to_json(parts, output_file)
        
        total_parts = len(parts)
        total_chapters = sum(len(part["chapters"]) for part in parts)
        
        print(f"Успешно извлечено {total_parts} частей и {total_chapters} глав")
        print(f"Сохранено в {output_file}")
        
        # Выводим структуру для проверки
        print_json_structure(output_file)
                
    except FileNotFoundError:
        print(f"Файл {input_file} не найден")
    except Exception as e:
        print(f"Произошла ошибка: {e}")