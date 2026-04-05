import spacy
import os
import json

nlp = spacy.load("en_core_web_trf")

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/scripts/filtered")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../data/knowledge_graph/entities")

def extract_entities(text):
    doc = nlp(text)
    seen = set()
    entities = []
    for ent in doc.ents:
        key = (ent.text, ent.label_)
        if key not in seen:
            seen.add(key)
            entities.append({"text": ent.text, "label": ent.label_})
    return entities

def check_coverage():
    scripts = {os.path.splitext(f)[0] for f in os.listdir(DATA_DIR) if f.endswith(".txt")}
    entities = {f.replace("_entities.json", "") for f in os.listdir(OUTPUT_DIR) if f.endswith("_entities.json")}
    missing = scripts - entities
    extra = entities - scripts
    if missing:
        print(f"[check] Missing entities for ({len(missing)}): {sorted(missing)}")
    if extra:
        print(f"[check] Extra entity files with no matching script ({len(extra)}): {sorted(extra)}")
    if not missing and not extra:
        print(f"[check] OK — {len(scripts)} scripts, {len(entities)} entity files. All match.")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(DATA_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            entities = extract_entities(text)
            output = {
                "file": filename,
                "entities": entities
            }
            out_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}_entities.json")
            with open(out_path, "w", encoding="utf-8") as out_f:
                json.dump(output, out_f, ensure_ascii=False, indent=2)
            print(f"Processed {filename}, found {len(entities)} entities.")
    check_coverage()

if __name__ == "__main__":
    main()
