import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

ENTITIES_DIR = os.path.join(os.path.dirname(__file__), "../data/knowledge_graph/entities_filtered")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../data/knowledge_graph/entities_clean")

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL   = os.getenv("LLM_MODEL")
LLM_API_URL = os.getenv("LLM_API_URL")

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_URL)


def call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=16384,
    )
    return response.choices[0].message.content


def llm_filter(entities: list[dict]) -> list[dict]:
    entity_list = "\n".join(f'- {e["text"]} ({e["label"]})' for e in entities)
    prompt = f"""You are cleaning a named entity list extracted from a film screenplay.

Your task: remove any entries that are NOT real named entities. Keep only:
- Character names (real people or fictional characters)
- Real locations (cities, countries, regions, buildings)
- Real organizations (companies, institutions, teams)
- Real works of art (films, books, songs)
- Nationalities, religions, political groups

Remove anything that is:
- A screenplay direction or annotation (e.g. VOICE, CAMERA, ANGLE, POV)
- A scene heading fragment (e.g. NIGHT DAVE, DAY MR)
- A role descriptor without a name (e.g. OLDER WOMAN, POLITICIAN, BUTLER, DRIVER)
- A common word or verb mistakenly tagged (e.g. GRABS, FUCK, BEGINNING)
- An incomplete or garbled fragment (e.g. NGLE, DBL, P.I)

Return ONLY a valid JSON array of the kept entities, same format as input. No explanation, no markdown.
Do NOT modify, rename, or correct any entity text — keep the exact text and label as given. Only remove entries.

Entities:
{entity_list}

Output format:
[{{"text": "Entity Name", "label": "PERSON"}}]
"""
    raw = call_llm(prompt)
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())
    return json.loads(raw)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = [f for f in os.listdir(ENTITIES_DIR) if f.endswith("_entities.json")]
    total = len(files)
    for i, filename in enumerate(files, 1):
        in_path = os.path.join(ENTITIES_DIR, filename)
        out_path = os.path.join(OUTPUT_DIR, filename)

        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        original = data.get("entities", [])
        try:
            cleaned = llm_filter(original)
        except Exception as e:
            print(f"[{i}/{total}] {filename}: error — {e}, keeping original")
            cleaned = original

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"file": data["file"], "entities": cleaned}, f, ensure_ascii=False, indent=2)

        print(f"[{i}/{total}] {filename}: {len(original)} → {len(cleaned)} entities")


if __name__ == "__main__":
    main()
