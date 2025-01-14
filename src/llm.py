import ollama
import numpy as np
from pydantic import BaseModel
from typing import Generator


MODEL = 'llama3.3'
PROMPT_TEMPLATE = """You are designated as an assistant that identify and extract topics from reviews of video games on Steam.
You are asked to provide a list of topics and you must report the citations that support each topic.
You can reformulate the citations as needed. If a review is trivial or does not contain any topic, you can skip it.
Respond using JSON.

## Reviews:
{reviews}

## Topics:"""


class TopicInfo(BaseModel):
    topic: str
    citations: list[str]


class TopicList(BaseModel):
    topics: list[TopicInfo]


def get_prompt(reviews: list[str]) -> str:
    return PROMPT_TEMPLATE.format(reviews='\n\n'.join(reviews))


def get_topics(prompt: str) -> TopicList:
    response = ollama.generate(
        model=MODEL,
        prompt=prompt,
        format=TopicList.model_json_schema(),
        options={'temperature': 0.2}
    )

    # Regex to clean reviews (.)\1{20,}
    return TopicList.model_validate_json(response.response)


def get_batch(reviews: list, batch_size: int = 16) -> Generator[list, None, None]:
    for i in range(0, len(reviews), batch_size):
        yield reviews[i:i + batch_size]


if __name__ == '__main__':
    import sys
    import os
    import pandas as pd
    from tqdm.auto import tqdm

    os.makedirs('output', exist_ok=True)
    os.makedirs('output/topics', exist_ok=True)

    game = sys.argv[1].split('/')[-1].split('.')[0]

    df = pd.read_csv(os.path.join(os.getcwd(), sys.argv[1]))
    reviews = df['review'].astype(str).tolist()

    topics = []
    batch_size = 16
    for batch in tqdm(get_batch(reviews, batch_size=batch_size), desc='Processing', total=len(reviews)//batch_size):
        prompt = get_prompt(batch)
        response = get_topics(prompt)
        topics.extend([t.model_dump() for t in response.topics])
    
    df_topics = pd.DataFrame(topics)
    all_topics = df_topics['topic'].tolist()
    df_topics.explode('citations').to_csv(f'output/topics/{game}_topics.csv', index=False)
