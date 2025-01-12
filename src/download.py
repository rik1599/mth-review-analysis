import os
import requests
import json
import html
from tqdm.auto import tqdm
from urllib.parse import quote
import pandas as pd

URL = 'https://www.togeproductions.com/SteamScout/proxy.php?appid={steam_id}&filter=recent&language={lang}&day_range=null&cursor={cursor}&review_type=all&purchase_type=all&num_per_page=100&filter_offtopic_activity=null'


def download_page(steam_id: str, cursor: str='*', lang: str='all') -> dict:
    os.makedirs('./data/reviews', exist_ok=True)

    # First request
    cursor = quote(cursor, safe=':/*')
    url = URL.format(steam_id=steam_id, cursor=cursor, lang=lang)
    r = requests.get(url)
    text = html.unescape(r.text)
    return json.loads(text)


def get_reviews(data: dict) -> dict:
    return [
        {
            'language': review['language'],
            'review': review['review'],
        }
        for review in data['reviews']
    ]


def download_reviews(steam_id: str) -> None:
    page = download_page(steam_id)
    n_pages = page['query_summary']['total_reviews'] // 100
    cursor = page['cursor']
    reviews = get_reviews(page)

    for _ in tqdm(range(n_pages), leave=False):
        page = download_page(steam_id=steam_id, cursor=cursor)
        cursor = page['cursor']
        reviews += get_reviews(page)

    df = pd.DataFrame(reviews)
    df.to_csv(f'./data/reviews/{steam_id}.csv', index=False)


if __name__ == '__main__':
    games = pd.read_csv('./Game List - Final.csv')
    games = games.dropna(subset=['SteamID'])
    games = games[games['Final decision'] == 'Yes']
    
    print(f"Total games: {len(games)}")
    games['SteamID'] = games['SteamID'].astype(int).astype(str)

    for steam_id in (bar := tqdm(games['SteamID'])):
        bar.set_description(steam_id)
        download_reviews(steam_id)