import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/scripts/filtered")
ENTITIES_DIR = os.path.join(os.path.dirname(__file__), "../data/knowledge_graph/entities_clean")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../data/knowledge_graph/relations")

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL   = os.getenv("LLM_MODEL")
LLM_API_URL = os.getenv("LLM_API_URL")

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_URL)

MAX_CHUNK_CHARS = 4000

# Character-to-character predicates
CHAR_CHAR_PREDICATES = {
    "BETRAYS", "TEACHES", "SAVES", "LIES_TO", "CONFRONTS", "ALLIES_WITH",
    "THREATENS", "RECONCILES_WITH", "KILLS", "AVENGES", "MANIPULATES",
    "PROTECTS", "SACRIFICES_FOR", "OWES", "LOVES", "HATES", "FORGIVES",
    "BLAMES", "ABANDONS", "ENVIES",
}

# Character-to-concept predicates
CHAR_CONCEPT_PREDICATES = {
    "WANTS", "LEARNS", "LOSES", "DISCOVERS", "BELIEVES", "DOUBTS",
    "FEARS", "REVEALS", "REJECTS", "MASTERS", "INHERITS", "SEEKS",
    "ACCEPTS", "POSSESSES", "DESTROYS", "CREATES",
}

# Transformation predicates
TRANSFORMATION_PREDICATES = {
    "TRANSFORMS_INTO", "CORRUPTS", "REDEEMS", "REALIZES",
}

# Social/family predicates
SOCIAL_PREDICATES = {
    "MARRIED_TO", "PARENT_OF", "CHILD_OF", "SIBLINGS_WITH", "WORKS_FOR", "LEADS",
}

VALID_PREDICATES = CHAR_CHAR_PREDICATES | CHAR_CONCEPT_PREDICATES | TRANSFORMATION_PREDICATES | SOCIAL_PREDICATES


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


ENTITY_TYPES = {"PERSON", "EVENT", "ORG", "NORP", "GPE", "LOC", "FAC", "WORK_OF_ART"}


def filter_entities_for_relations(entities: list[dict], filename: str = "") -> list[dict]:
    malformed = [e for e in entities if not isinstance(e, dict)]
    if malformed:
        print(f"  WARNING: {filename} has {len(malformed)} malformed entries (not dicts), skipping them.")
    return [e for e in entities if isinstance(e, dict) and e.get("label") in ENTITY_TYPES]


def extract_relations(chunk: str, entities: list[dict]) -> list[dict]:
    entity_list = "\n".join(f'- {e["text"]} ({e["label"]})' for e in entities)
    canonical_names = ", ".join(sorted({e["text"] for e in entities}))
    predicates = ", ".join(sorted(VALID_PREDICATES))
    prompt = f"""You are an information extraction assistant building a knowledge graph for films.

Given the following text and named entities, extract meaningful relations between entities.
Only extract relations clearly supported by the text. Do not infer or hallucinate.
If no relations from the predicate list are clearly supported, return an empty array [].

You MUST use only these predicates: {predicates}

"from" must always be a PERSON. "to" can be any entity type.

IMPORTANT — use ONLY these canonical entity names (exact spelling, exact case):
{canonical_names}
Do not invent names, do not use pronouns, do not use name variants.

The "evidence" field must contain a direct quote from the text showing a clear action or statement that supports the predicate. A mere mention of both entities is not sufficient evidence.

Return ONLY a valid JSON array. No explanation, no markdown, just JSON.
Each item must have exactly these keys: "from", "from_type", "to", "to_type", "label", "evidence".

Entities:
{entity_list}

Text:
{chunk}

Output format:
[{{"from": "Entity A", "from_type": "PERSON", "to": "Entity B", "to_type": "PERSON", "label": "BETRAYS", "evidence": "direct quote from text"}}]
"""
    raw = call_llm(prompt)
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())
    raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", raw)
    relations = json.loads(raw)
    canonical = {e["text"] for e in entities}
    return [
        r for r in relations
        if r.get("label") in VALID_PREDICATES
        and r.get("from_type") == "PERSON"
        and r.get("from") in canonical
        and r.get("to") in canonical
    ]


def deduplicate_relations(relations: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for r in relations:
        key = (r["from"], r["label"], r["to"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def main(limit=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = [f for f in os.listdir(ENTITIES_DIR) if f.endswith("_entities.json")]
    if limit:
        files = files[:limit]

    for filename in files:

        base = filename.replace("_entities.json", "")
        entities_path = os.path.join(ENTITIES_DIR, filename)
        script_path = os.path.join(DATA_DIR, f"{base}.txt")
        out_path = os.path.join(OUTPUT_DIR, f"{base}_relations.json")

        if not os.path.exists(script_path):
            print(f"Skipping {base}: script not found.")
            continue

        if os.path.exists(out_path):
            print(f"Skipping {base}: already processed.")
            continue

        with open(entities_path, "r", encoding="utf-8") as f:
            entities_data = json.load(f)
        entities = filter_entities_for_relations(entities_data.get("entities", []), filename)

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
