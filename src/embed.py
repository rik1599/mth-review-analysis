import ollama
import numpy as np
from enum import Enum

MODEL='nomic-embed-text'

class NomicPrefix(Enum):
    DOCUMENT = 'search_document'
    QUERY = 'search_query'
    CLUSTERING = 'clustering'
    CLASSIFICATION = 'classification'

def get_embedding(text: list[str], prefix: NomicPrefix = NomicPrefix.DOCUMENT) -> np.ndarray:
    emb = ollama.embed(
        model=MODEL,
        input=[f"{prefix.value}: {t}" for t in text],
    ).embeddings
    return np.array(emb)

if __name__ == '__main__':
    import sys
    import os
    import pandas as pd
    from tqdm.auto import tqdm

    os.makedirs('output', exist_ok=True)
    os.makedirs('output/embeddings', exist_ok=True)
    os.makedirs('output/embeddings/topics', exist_ok=True)
    
    game = os.path.basename(sys.argv[1]).split('.')[0]
    df = pd.read_csv(sys.argv[1])
    df.rename(columns={'citations': 'review'}, inplace=True)

    reviews = df['review'].astype(str).tolist()

    df['topic'] = df['topic'].astype(str).apply(lambda x: x.lower())
    topics = df['topic'].unique().tolist()

    BATCH_SIZE = 128
    CTX_SIZE = get_embedding(['test']).shape[1]
    reviews_emb = np.zeros((len(reviews), CTX_SIZE))
    topics_emb = np.zeros((len(topics), CTX_SIZE))

    for i in tqdm(range(0, len(reviews), BATCH_SIZE), desc='Processing reviews'):
        reviews_emb[i:i+BATCH_SIZE] = get_embedding(reviews[i:i+BATCH_SIZE])
    
    for i in tqdm(range(0, len(topics), BATCH_SIZE), desc='Processing topics'):
        topics_emb[i:i+BATCH_SIZE] = get_embedding(topics[i:i+BATCH_SIZE])
    
    df['embedding'] = reviews_emb.tolist()
    df.to_parquet(f'output/embeddings/{game}.parquet', index=False, compression='gzip')

    df_topics = pd.DataFrame(data=topics_emb, index=topics)
    df_topics.to_parquet(f'output/embeddings/topics/{game}.parquet', index=True, compression='gzip')
