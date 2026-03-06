import os, sys
from transformers import pipeline as hf_pipeline
from sqlalchemy import text
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from db.db_utils import get_engine

print('Cargando modelo FinBERT...')
finbert = hf_pipeline(
    task='text-classification',
    model='ProsusAI/finbert',
    top_k=None,        # devolver las 3 etiquetas con sus scores
    truncation=True,
    max_length=512
)
print('Modelo cargado.')

def scores_from_result(result):
    mapping = {item['label'].lower(): item['score'] for item in result}
    pos = mapping.get('positive', 0.0)
    neg = mapping.get('negative', 0.0)
    neu = mapping.get('neutral',  0.0)
    return {'score_positive': round(pos, 4),
            'score_negative': round(neg, 4),
            'score_neutral':  round(neu, 4),
            'compound_score': round(pos - neg, 4)}

def get_unprocessed_texts(batch_size=200):
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, text FROM raw_texts
            WHERE processed = FALSE
            ORDER BY timestamp ASC LIMIT :batch_size"""),
            {'batch_size': batch_size})
        return [{'id': r[0], 'text': r[1]} for r in rows]

def mark_as_processed(text_ids):
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(
            'UPDATE raw_texts SET processed=TRUE WHERE id=ANY(:ids)'),
            {'ids': text_ids})

def save_scores(scores_list):
    engine = get_engine()
    with engine.begin() as conn:
        for s in scores_list:
            conn.execute(text("""
                INSERT INTO sentiment_scores
                (text_id,score_positive,score_negative,score_neutral,
                 compound_score,model_used)
                VALUES (:text_id,:score_positive,:score_negative,
                        :score_neutral,:compound_score,'ProsusAI/finbert')"""), s)

def process_all_texts(batch_size=200):
    total = 0
    while True:
        batch = get_unprocessed_texts(batch_size)
        if not batch: print('No quedan textos por procesar.'); break
        texts    = [item['text'] for item in batch]
        text_ids = [item['id']   for item in batch]
        texts_trunc = [t[:1000] for t in texts]
        results = finbert(texts_trunc, batch_size=32)
        scores_list = []
        for text_id, result in zip(text_ids, results):
            s = scores_from_result(result)
            s['text_id'] = text_id
            scores_list.append(s)
        save_scores(scores_list)
        mark_as_processed(text_ids)
        total += len(batch)
        print(f'  Procesados {total} textos...')
    return total

if __name__ == '__main__':
    n = process_all_texts(batch_size=200)
    print(f'Total procesado: {n} textos')
