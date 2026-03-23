import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "../fede/data/scripts/filtered")
ENTITIES_DIR = os.path.join(os.path.dirname(__file__), "../fede/data/entities")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../fede/data/relations")

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL   = os.getenv("LLM_MODEL")
LLM_API_URL = os.getenv("LLM_API_URL")

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_URL)

MAX_CHUNK_CHARS = 4000


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    paragraphs = text.split("\n\n")
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) > max_chars and current:
            chunks.append(current.strip())
            current = para
        else:
            current += "\n\n" + para
    if current.strip():
        chunks.append(current.strip())
    return chunks


def call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
    )
    return response.choices[0].message.content


def extract_relations(chunk: str, entities: list[dict]) -> list[dict]:
    entity_list = "\n".join(f'- {e["text"]} ({e["label"]})' for e in entities)
    prompt = f"""You are an information extraction assistant building a knowledge graph.

Given the following text and the named entities found in it, extract meaningful relations between entities.
Only extract relations that are clearly supported by the text. Do not infer or hallucinate.

Return ONLY a valid JSON array of triples. No explanation, no markdown, just JSON.
Each triple must have exactly these keys: "subject", "relation", "object".

Entities:
{entity_list}

Text:
{chunk}

Output format:
[{{"subject": "Entity A", "relation": "relation_type", "object": "Entity B"}}]
"""
    raw = call_llm(prompt)
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())
    return json.loads(raw)


def deduplicate_relations(relations: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for r in relations:
        key = (r["subject"], r["relation"], r["object"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for filename in os.listdir(ENTITIES_DIR):
        if not filename.endswith("_entities.json"):
            continue

        base = filename.replace("_entities.json", "")
        entities_path = os.path.join(ENTITIES_DIR, filename)
        script_path = os.path.join(DATA_DIR, f"{base}.txt")
        out_path = os.path.join(OUTPUT_DIR, f"{base}_relations.json")

        if not os.path.exists(script_path):
            print(f"Skipping {base}: script not found.")
            continue

        with open(entities_path, "r", encoding="utf-8") as f:
            entities_data = json.load(f)
        entities = entities_data.get("entities", [])

        with open(script_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        all_relations = []

        for i, chunk in enumerate(chunks):
            print(f"  chunk {i+1}/{len(chunks)}...", end=" ", flush=True)
            try:
                relations = extract_relations(chunk, entities)
                all_relations.extend(relations)
                print(f"{len(relations)} relations found.")
            except Exception as e:
                print(f"error: {e}")

        all_relations = deduplicate_relations(all_relations)

        output = {"file": f"{base}.txt", "relations": all_relations}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"[{base}] {len(all_relations)} unique relations saved.")


if __name__ == "__main__":
    main()
