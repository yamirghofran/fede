import os
import re
import json

ENTITIES_DIR = os.path.join(os.path.dirname(__file__), "../data/knowledge_graph/entities")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../data/knowledge_graph/entities_filtered")

KEEP_LABELS = {"PERSON", "GPE", "LOC", "FAC", "ORG", "EVENT", "WORK_OF_ART", "NORP"}
# Filtering: CARDINAL, ORDINAL, QUANTITY, PERCENT, MONEY, TIME, DATE, LANGUAGE - less relevant for relation extraction. 

SCREENPLAY_ANNOTATIONS = re.compile(
    r'\b(V\.O\.?|O\.S\.?|CONT\'?D|PRELAP|VISUALS|INTERCUT|PRE-LAP)\b', re.IGNORECASE
)
SCENE_HEADING = re.compile(r'^[A-Z\s\'\-]+\s+-\s+(DAY|NIGHT|CONTINUOUS|LATER|MORNING|EVENING|DUSK|DAWN|AFTERNOON|DUSK)$')
SCENE_HEADING_PREFIX = re.compile(r'^(INT|EXT|I/E|E/I)([\./\s]|$)', re.IGNORECASE)
SINGLE_ARTICLES = {"THE", "A", "AN"}
STARTS_WITH_DIGIT = re.compile(r'^\d')
POSSESSIVE = re.compile(r"[''\u2019]s?$", re.IGNORECASE)
TRAILING_JUNK = re.compile(r'[-–—]+$')
PARENTHETICAL = re.compile(r'\s*\([^)]*\)\s*')
STUTTER = re.compile(r'^(\w+)-\1-', re.IGNORECASE)
ENDS_WITH_PREPOSITION = re.compile(r'\b(of|the|a|an|and|or|in|on|at|to|for|with|by)$', re.IGNORECASE)
TIME_OF_DAY = {"DAY", "NIGHT", "DAWN", "DUSK", "MORNING", "EVENING", "AFTERNOON", "LATER", "CONTINUOUS", "FLASH", "FLASHBACK"}
SCREENPLAY_DIRECTIONS = {"VOICE", "CAMERA", "ANGLE", "CUT", "FADE", "SMASH", "POV", "CLOSE", "WIDE", "PULL", "PUSH", "RACK"}


def normalize_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = PARENTHETICAL.sub(" ", text).strip()
    text = SCREENPLAY_ANNOTATIONS.sub("", text).strip(" .()")
    text = TRAILING_JUNK.sub("", text).strip()
    text = POSSESSIVE.sub("", text).strip()
    return text


def is_noise(text: str) -> bool:
    if len(text) <= 1:
        return True
    if STARTS_WITH_DIGIT.match(text):
        return True
    if SCENE_HEADING.match(text.upper()):
        return True
    if SCENE_HEADING_PREFIX.match(text):
        return True
    if text.upper() in TIME_OF_DAY:
        return True
    if text.upper() in SINGLE_ARTICLES:
        return True
    if text.islower() and " " not in text:
        return True
    if any(c in text for c in ['"', '\\']):
        return True
    if STUTTER.match(text):
        return True
    if ENDS_WITH_PREPOSITION.search(text):
        return True
    if text.upper() in SCREENPLAY_DIRECTIONS:
        return True
    return False


def clean_entities(entities: list[dict]) -> list[dict]:
    # label filter + normalize + noise filter
    cleaned = []
    for e in entities:
        if e["label"] not in KEEP_LABELS:
            continue
        text = normalize_text(e["text"])
        if not text or is_noise(text):
            continue
        cleaned.append({"text": text, "label": e["label"]})

    # deduplicate
    groups: dict[str, dict] = {}
    for e in cleaned:
        key = e["text"].lower()
        if key not in groups:
            groups[key] = e
        else:
            existing = groups[key]["text"]
            candidate = e["text"]
            if existing.isupper() and not candidate.isupper():
                groups[key] = e

    return list(groups.values())


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = [f for f in os.listdir(ENTITIES_DIR) if f.endswith("_entities.json")]
    total = len(files)
    for i, filename in enumerate(files, 1):
        in_path = os.path.join(ENTITIES_DIR, filename)
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        original = data.get("entities", [])
        filtered = clean_entities(original)

        out_path = os.path.join(OUTPUT_DIR, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"file": data["file"], "entities": filtered}, f, ensure_ascii=False, indent=2)

        print(f"[{i}/{total}] {filename}: {len(original)} -> {len(filtered)} entities")


if __name__ == "__main__":
    main()


