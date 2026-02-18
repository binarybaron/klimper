# Klimper

![image](https://github.com/user-attachments/assets/54f7100f-3be3-4dd4-a82e-f2daf2d78ed6)


Generate Anki flashcards with music trivia from your Spotify playlists.

For each song, Klimper fetches context from Wikipedia, generates grounded Q&A pairs via an LLM, and packages them into an Anki deck with album art and 30-second song previews.

## Setup

### 1. Export your Spotify playlists

Go to https://exportify.net/, log in with Spotify, and export playlists as CSV files. Place them in a `playlists/` directory:

```
playlists/
  My_Playlist.csv
  Another_Playlist.csv
```

### 2. Get an OpenRouter API key

Sign up at https://openrouter.ai/ and create an API key. Add it to a `.env` file:

```
OPENROUTER_API_KEY=sk-or-...
```

### 3. Install dependencies

```
uv sync
```

## Usage

```
uv run klimper.py [--limit N] [--no-cache] [--output FILE]
```

The tool will prompt you to select a playlist, then run through three stages:

1. **Wikipedia enrichment** -- searches for context about each song/artist
2. **Fact generation** -- generates German-language Q&A pairs grounded in Wikipedia
3. **Deck creation** -- builds an `.apkg` file with album art and audio previews

### Options

- `--limit N` -- only process the first N songs (useful for testing)
- `--no-cache` -- ignore cached results and re-fetch everything
- `--output FILE` -- custom output path (default: auto-generated from playlist name and date)

### Output

The generated `.apkg` file can be imported directly into Anki. Each card includes:

- A trivia question (front)
- Album cover art
- 30-second song preview
- Answer with Wikipedia source link (back)

## Caching

All API responses, album art, and audio previews are cached in `cache/`. Delete this directory (or use `--no-cache`) to force a full refresh.
