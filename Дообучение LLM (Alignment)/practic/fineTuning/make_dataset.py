import json
import re
from tqdm import tqdm
from nltk.stem.snowball import SnowballStemmer

input_file = "output/qa_results.json"
output_file = "output/dataset.jsonl"

stemmer = SnowballStemmer("russian")

def extract_keywords(question):
    words = re.findall(r'\w+', question.lower())
    return [stemmer.stem(w) for w in words if len(w) > 2]

def filter_text(text, keywords, max_sentences=3):
    # Разбиваем текст на предложения через регулярку
    sentences = re.split(r'(?<=[.!?])\s+', text)
    scored_sentences = []
    
    for s in sentences:
        words = [stemmer.stem(w) for w in re.findall(r'\w+', s.lower())]
        score = sum(1 for k in keywords if k in words)
        if score > 0:
            scored_sentences.append((score, s))
    
    if scored_sentences:
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        filtered = [s for _, s in scored_sentences[:max_sentences]]
        return " ".join(filtered)
    else:
        return " ".join(sentences[:2])

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

with open(output_file, 'w', encoding='utf-8') as f_out:
    for item in tqdm(data, desc="Processing items"):
        questions = item.get("questions", [])
        answers = item.get("answers", [item.get("source_chunk", "")] * len(questions))
        
        for q, a in zip(questions, answers):
            keywords = extract_keywords(q)
            short_answer = filter_text(a, keywords)
            example = {
                "request": [{"role": "user", "text": q}],
                "response": short_answer
            }
            f_out.write(json.dumps(example, ensure_ascii=False) + "\n")
