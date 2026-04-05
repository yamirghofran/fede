import os
import json

ENTITIES_DIR = os.path.join(os.path.dirname(__file__), "../data/knowledge_graph/entities")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../data/knowledge_graph/entities_filtered")


KEEP_LABELS = {"PERSON", "GPE", "LOC", "FAC", "ORG", "EVENT", "WORK_OF_ART", "NORP"}
# Filtering: CARDINAL, ORDINAL, QUANTITY, PERCENT, MONEY, TIME, DATE, LANGUAGE - less relevant for relation extraction. 

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = [f for f in os.listdir(ENTITIES_DIR) if f.endswith("_entities.json")]
    total = len(files)
    for i, filename in enumerate(files, 1):
        in_path = os.path.join(ENTITIES_DIR, filename)
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        original = data.get("entities", [])
        filtered = [e for e in original if e["label"] in KEEP_LABELS]

        out_path = os.path.join(OUTPUT_DIR, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"file": data["file"], "entities": filtered}, f, ensure_ascii=False, indent=2)

        print(f"[{i}/{total}] {filename}: {len(original)} -> {len(filtered)} entities")


if __name__ == "__main__":
    main()
