from enum import Enum
import ollama
from pydantic import BaseModel

MODEL = 'phi4'
PROMPT = """Classify the text into Very Negative, Negative, Neutral, Positive, or Very Positive. Respond using JSON format.
Text: {input}
Sentiment: """

class Sentiment(str, Enum):
    VERY_NEGATIVE = 'Very Negative'
    NEGATIVE = 'Negative'
    NEUTRAL = 'Neutral'
    POSITIVE = 'Positive'
    VERY_POSITIVE = 'Very Positive'

class SentimentResponse(BaseModel):
    sentiment: Sentiment

def sentiment(prompt: str):
    response = ollama.generate(
        model=MODEL,
        prompt=PROMPT.format(input=prompt),
        options={'temperature': 0},
        format=SentimentResponse.model_json_schema()
    ).response
    
    return SentimentResponse.model_validate_json(response)


if __name__ == '__main__':
    import pandas as pd
    import sys
    import os
    from tqdm.auto import tqdm

    os.makedirs('output', exist_ok=True)
    os.makedirs('output/sentiments', exist_ok=True)

    game = sys.argv[1].split('/')[-1].split('.')[0]
    df = pd.read_csv(sys.argv[1])

    citations = df['citations'].astype(str).tolist()
    sentiments = [
        sentiment(citation).sentiment.value
        for citation in tqdm(citations, desc='Processing sentiments')
    ]

    sentiments = pd.Series(sentiments)
    sentiments.to_csv(os.path.join('output', 'sentiments', f'{game}_sentiments.csv'), index=False)