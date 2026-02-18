"""Klimper — Anki flashcard generator for Spotify liked songs."""

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import time
from datetime import date
from pathlib import Path

import dotenv
import genanki
import requests
from openai import OpenAI
from tqdm import tqdm

CACHE_DIR = Path("cache")
ART_DIR = Path("cache/artwork")
PREVIEW_DIR = Path("cache/previews")
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {"User-Agent": "Klimper/1.0 (Anki flashcard generator; Python/requests)"}
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MODEL = "anthropic/claude-3.5-sonnet"


def pick_csv() -> str:
    """Prompt user to select a playlist CSV from the playlists/ directory."""
    playlist_dir = Path("playlists")
    if not playlist_dir.is_dir():
        print("No playlists/ directory found.")
        raise SystemExit(1)
    csvs = sorted(playlist_dir.glob("*.csv"))
    if not csvs:
        print("No CSV files found in playlists/.")
        raise SystemExit(1)
    if len(csvs) == 1:
        print(f"Using {csvs[0]}")
        return str(csvs[0])
    print("Available playlists:")
    for i, p in enumerate(csvs, 1):
        print(f"  {i}) {p.stem.replace('_', ' ')}")
    while True:
        choice = input(f"Select playlist [1-{len(csvs)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(csvs):
            return str(csvs[int(choice) - 1])
        print("Invalid choice, try again.")


def load_songs(path: str, limit: int | None = None) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        songs = list(reader)
    if limit:
        songs = songs[:limit]
    return songs


def cache_key(song: dict) -> str:
    uri = song["Track URI"]
    return hashlib.sha256(uri.encode()).hexdigest()[:16]


def load_cache(key: str, suffix: str) -> dict | None:
    path = CACHE_DIR / f"{key}_{suffix}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def save_cache(key: str, suffix: str, data: dict):
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{key}_{suffix}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def llm_cache_key(model: str, prompt: str) -> str:
    """Cache key based on model + prompt content."""
    blob = f"{model}\n{prompt}"
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def llm_cached_call(client: OpenAI, model: str, prompt: str, use_cache: bool = True) -> str:
    """Call LLM with caching based on model + input."""
    key = llm_cache_key(model, prompt)
    if use_cache:
        cached = load_cache(key, "llm")
        if cached is not None:
            return cached["response"]

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content.strip()

    save_cache(key, "llm", {"model": model, "response": text})
    return text


# --- Album Art (iTunes) ---


def fetch_album_art(song: dict, use_cache: bool = True) -> str | None:
    """Fetch album artwork via iTunes Search API. Returns local file path or None."""
    key = cache_key(song)
    art_path = ART_DIR / f"{key}.jpg"
    miss_path = ART_DIR / f"{key}.miss"

    if use_cache:
        if art_path.exists():
            return str(art_path)
        if miss_path.exists():
            return None

    artist = song["Artist Name(s)"].split(";")[0]  # first artist only
    album = song["Album Name"]
    query = f"{artist} {album}"

    try:
        resp = requests.get(
            "https://itunes.apple.com/search",
            params={"term": query, "media": "music", "entity": "album", "limit": 1},
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            ART_DIR.mkdir(parents=True, exist_ok=True)
            miss_path.touch()
            return None

        art_url = results[0].get("artworkUrl100", "")
        if not art_url:
            ART_DIR.mkdir(parents=True, exist_ok=True)
            miss_path.touch()
            return None

        # Get higher resolution (600x600)
        art_url = art_url.replace("100x100bb", "600x600bb")

        img_resp = requests.get(art_url, timeout=15)
        img_resp.raise_for_status()

        ART_DIR.mkdir(parents=True, exist_ok=True)
        art_path.write_bytes(img_resp.content)
        return str(art_path)
    except Exception:
        ART_DIR.mkdir(parents=True, exist_ok=True)
        miss_path.touch()
        return None


# --- Song Preview (iTunes) ---


def download_with_retry(url: str, dest: Path, max_retries: int = 3) -> bool:
    """Download a URL to a file with exponential backoff."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)
            return True
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return False


def fetch_song_preview(song: dict, use_cache: bool = True) -> str | None:
    """Fetch 30s song preview via iTunes Search API. Returns local file path or None."""
    key = cache_key(song)
    preview_path = PREVIEW_DIR / f"{key}.m4a"
    miss_path = PREVIEW_DIR / f"{key}.miss"

    if use_cache:
        if preview_path.exists():
            return str(preview_path)
        if miss_path.exists():
            return None

    artist = song["Artist Name(s)"].split(";")[0]
    track = song["Track Name"]
    query = f"{artist} {track}"

    try:
        resp = requests.get(
            "https://itunes.apple.com/search",
            params={"term": query, "media": "music", "entity": "song", "limit": 1},
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
            miss_path.touch()
            return None

        preview_url = results[0].get("previewUrl", "")
        if not preview_url:
            PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
            miss_path.touch()
            return None

        if download_with_retry(preview_url, preview_path):
            return str(preview_path)

        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        miss_path.touch()
        return None
    except Exception:
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        miss_path.touch()
        return None


# --- Stage 1: Wikipedia Enrichment ---


def wiki_search(query: str) -> int | None:
    """Search Wikipedia and return the page ID of the first result, or None."""
    resp = requests.get(
        WIKI_API,
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 1,
            "format": "json",
        },
        headers=WIKI_HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json().get("query", {}).get("search", [])
    if results:
        return results[0]["pageid"]
    return None


def wiki_extract(page_id: int) -> tuple[str, str]:
    """Fetch the full plain-text extract and canonical URL for a page."""
    resp = requests.get(
        WIKI_API,
        params={
            "action": "query",
            "prop": "extracts|info",
            "exintro": "false",
            "explaintext": "true",
            "inprop": "url",
            "pageids": page_id,
            "format": "json",
        },
        headers=WIKI_HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    page = resp.json()["query"]["pages"][str(page_id)]
    return page.get("extract", ""), page.get("fullurl", "")


def fetch_wiki(song: dict, use_cache: bool = True) -> dict | None:
    """Fetch Wikipedia content for a song with fallback search strategies."""
    key = cache_key(song)
    if use_cache:
        cached = load_cache(key, "wiki")
        if cached is not None:
            return cached if cached.get("extract") else None

    track = song["Track Name"]
    artist = song["Artist Name(s)"]
    album = song["Album Name"]

    queries = [
        f'"{track}" {artist}',
        f"{track} {artist} song",
        f'"{album}" {artist}',
        f"{artist}",
    ]

    for query in queries:
        page_id = wiki_search(query)
        time.sleep(1)  # rate limit
        if page_id:
            extract, url = wiki_extract(page_id)
            time.sleep(1)
            if extract and len(extract) > 200:
                result = {"extract": extract, "url": url, "query": query}
                save_cache(key, "wiki", result)
                return result

    save_cache(key, "wiki", {"extract": "", "url": "", "query": ""})
    return None


# --- Stage 2: Fact Generation ---


def generate_facts(
    song: dict, wiki_data: dict, client: OpenAI, use_cache: bool = True
) -> list[dict]:
    """Generate Q&A pairs from Wikipedia content using Claude via OpenRouter."""
    key = cache_key(song)
    if use_cache:
        cached = load_cache(key, "facts")
        if cached is not None:
            return cached.get("facts", [])

    prompt = f"""You are creating fun-fact trivia flashcards about music.

Here is a Wikipedia article excerpt:
---
{wiki_data['extract'][:8000]}
---

Song metadata:
- Title: {song['Track Name']}
- Artist: {song['Artist Name(s)']}
- Album: {song['Album Name']}
- Release Date: {song['Release Date']}
- Genres: {song.get('Genres', '')}

Generate 2-5 fun fact Q&A pairs about this song, artist, or album. Rules:
1. Every fact MUST be directly stated in or clearly supported by the Wikipedia text above.
2. Do NOT invent or assume any facts not in the text.
3. Make questions engaging and specific — avoid generic "who is the artist?" style.
4. Keep answers concise (1-3 sentences).
5. Write the questions in GERMAN. The answers should also be in GERMAN.
6. Do NOT include chart rankings, chart positions, or billboard/global ranking facts — these are not interesting.
7. Do NOT use exact dates. Rough years or decades are fine (e.g. "in den frühen 80ern" or "um 1992"), but not "am 14. März 1983".
8. DO focus on genre context — where the artist/song fits within a genre, what subgenre they pioneered or belong to, genre origins, influences, and connections to other movements.

Respond with ONLY a JSON array, no other text:
[{{"question": "...", "answer": "..."}}, ...]"""

    text = llm_cached_call(client, MODEL, prompt, use_cache)

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()

    try:
        facts = json.loads(text)
        facts = [f for f in facts if isinstance(f, dict) and "question" in f and "answer" in f]
    except (json.JSONDecodeError, TypeError):
        print(f"  Warning: Failed to parse LLM response for {song['Track Name']}")
        facts = []

    save_cache(key, "facts", {"facts": facts})
    return facts


# --- Stage 3: Anki Deck Creation ---

ANKI_MODEL = genanki.Model(
    1607392319,
    "Klimper Music Fact",
    fields=[
        {"name": "Question"},
        {"name": "Answer"},
        {"name": "Source"},
        {"name": "Song"},
        {"name": "Artist"},
        {"name": "AlbumArt"},
        {"name": "Preview"},
    ],
    templates=[
        {
            "name": "Music Fact",
            "qfmt": (
                '<div style="text-align:center;margin-bottom:12px;">{{AlbumArt}}</div>'
                '<div style="font-size:20px;">{{Question}}</div>'
                '<div style="margin-top:12px;color:#666;font-size:14px;">'
                "🎵 {{Song}} — {{Artist}}</div>"
                "{{Preview}}"
            ),
            "afmt": (
                "{{FrontSide}}<hr>"
                '<div style="font-size:18px;">{{Answer}}</div>'
                '<div style="margin-top:12px;font-size:12px;color:#888;">'
                "Source: {{Source}}</div>"
            ),
        }
    ],
)


def build_deck(all_facts: list[dict], output: str, deck_name: str):
    """Create an Anki deck from generated facts."""
    deck = genanki.Deck(2059400110, deck_name)
    media_files = []

    for entry in all_facts:
        song = entry["song"]
        source = entry.get("wiki_url", "")
        art_path = entry.get("art_path")
        preview_path = entry.get("preview_path")

        art_filename = Path(art_path).name if art_path else ""
        art_html = (
            f'<img src="{art_filename}" style="max-width:200px;border-radius:8px;">'
            if art_filename
            else ""
        )
        preview_filename = Path(preview_path).name if preview_path else ""
        preview_html = f"[sound:{preview_filename}]" if preview_filename else ""

        if art_path and art_path not in media_files:
            media_files.append(art_path)
        if preview_path and preview_path not in media_files:
            media_files.append(preview_path)

        for fact in entry["facts"]:
            if "question" not in fact or "answer" not in fact:
                continue
            note = genanki.Note(
                model=ANKI_MODEL,
                fields=[
                    fact["question"],
                    fact["answer"],
                    source,
                    song["Track Name"],
                    song["Artist Name(s)"],
                    art_html,
                    preview_html,
                ],
            )
            deck.add_note(note)

    pkg = genanki.Package(deck)
    pkg.media_files = media_files
    pkg.write_to_file(output)
    print(f"Wrote {len(deck.notes)} cards to {output} ({len(media_files)} album covers)")


# --- Main ---


def main():
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(description="Klimper: Anki flashcards from your liked songs")
    parser.add_argument("--limit", type=int, help="Only process first N songs")
    parser.add_argument("--no-cache", action="store_true", help="Skip cache, re-fetch everything")
    parser.add_argument("--output", help="Output file path (auto-generated if omitted)")
    args = parser.parse_args()

    use_cache = not args.no_cache

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        return

    client = OpenAI(base_url=OPENROUTER_BASE, api_key=api_key)

    csv_path = pick_csv()
    playlist_name = Path(csv_path).stem.replace("_", " ")
    songs = load_songs(csv_path, args.limit)
    print(f"Loaded {len(songs)} songs from '{playlist_name}'")

    # Stage 1a: Wikipedia enrichment
    print("\n--- Stage 1: Wikipedia Enrichment ---")
    wiki_results = {}
    wiki_miss = 0
    for song in tqdm(songs, desc="Fetching Wikipedia"):
        wiki = fetch_wiki(song, use_cache)
        if wiki:
            wiki_results[cache_key(song)] = wiki
        else:
            wiki_miss += 1

    print(f"Found Wikipedia content for {len(wiki_results)}/{len(songs)} songs ({wiki_miss} missed)")

    # Stage 1b: Album art + song previews
    print("\n--- Fetching Album Art & Previews ---")
    art_results = {}
    preview_results = {}
    for song in tqdm(songs, desc="Fetching artwork & previews"):
        art_path = fetch_album_art(song, use_cache)
        if art_path:
            art_results[cache_key(song)] = art_path
        preview_path = fetch_song_preview(song, use_cache)
        if preview_path:
            preview_results[cache_key(song)] = preview_path

    print(f"Found album art for {len(art_results)}/{len(songs)} songs")
    print(f"Found previews for {len(preview_results)}/{len(songs)} songs")

    # Stage 2: Fact generation (parallel)
    print("\n--- Stage 2: Fact Generation ---")
    all_facts = []
    songs_with_wiki = [s for s in songs if cache_key(s) in wiki_results]

    def _generate_one(song):
        key = cache_key(song)
        wiki = wiki_results[key]
        facts = generate_facts(song, wiki, client, use_cache)
        if facts:
            return {
                "song": song,
                "facts": facts,
                "wiki_url": wiki.get("url", ""),
                "art_path": art_results.get(key),
                "preview_path": preview_results.get(key),
            }
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_generate_one, s): s for s in songs_with_wiki}
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Generating facts",
        ):
            result = future.result()
            if result:
                all_facts.append(result)

    total_cards = sum(len(e["facts"]) for e in all_facts)
    print(f"Generated {total_cards} cards for {len(all_facts)} songs")

    # Stage 3: Anki deck creation
    print("\n--- Stage 3: Anki Deck Creation ---")
    today = date.today().isoformat()
    deck_name = f"Klimper: {playlist_name} ({len(songs)} songs, {today})"
    output = args.output or f"klimper_{Path(csv_path).stem}_{len(songs)}songs_{today}.apkg"
    build_deck(all_facts, output, deck_name)


if __name__ == "__main__":
    main()
