import ollama
import numpy as np

MODEL='nomic-embed-text'

def get_embedding(text: list[str]) -> np.ndarray:
    emb = ollama.embed(
        model=MODEL,
        input=[f"clustering: {t}" for t in text],
    ).embeddings
    return np.array(emb)

if __name__ == '__main__':
    import sys
    import os
    import pandas as pd
    from tqdm.auto import tqdm

    os.makedirs('output', exist_ok=True)
    os.makedirs('output/embeddings_topics', exist_ok=True)
    os.makedirs('output/embeddings_citations', exist_ok=True)

    game = sys.argv[1].split('/')[-1].split('.')[0]
    df = pd.read_csv(sys.argv[1])

    reviews = df['citations'].astype(str).tolist()
    topics = df['topic'].astype(str).unique().tolist()

    BATCH_SIZE = 128
    CTX_SIZE = get_embedding(['test']).shape[1]
    reviews_emb = np.zeros((len(reviews), CTX_SIZE))
    topics_emb = np.zeros((len(topics), CTX_SIZE))

    for i in tqdm(range(0, len(reviews), BATCH_SIZE), desc='Processing reviews'):
        reviews_emb[i:i+BATCH_SIZE] = get_embedding(reviews[i:i+BATCH_SIZE])
    
    for i in tqdm(range(0, len(topics), BATCH_SIZE), desc='Processing topics'):
        topics_emb[i:i+BATCH_SIZE] = get_embedding(topics[i:i+BATCH_SIZE])
    
    np.save(os.path.join('output', 'embeddings_topics', f'{game}_topics.npy'), topics_emb)
    pd.Series(topics).to_csv(os.path.join('output', 'embeddings_topics', f'{game}_topics.csv'), index=False)

    np.save(os.path.join('output', 'embeddings_citations', f'{game}_reviews.npy'), reviews_emb)
