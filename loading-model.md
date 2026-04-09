Loading the fine-tuned model (AnzeZ/fede-embeddinggemma)

The model on HuggingFace uses a LoRA adapter, which SentenceTransformer() doesn't load correctly on its own. You need to load the base model first, then apply the adapter with PEFT:

from sentence_transformers import SentenceTransformer
from peft import PeftModel
# Step 1: Load base model
model = SentenceTransformer("google/embeddinggemma-300m")
# Step 2: Apply the fine-tuned LoRA adapter
model[0].auto_model = PeftModel.from_pretrained(
    model[0].auto_model, "AnzeZ/fede-embeddinggemma"
)
# Step 3: Encode — always use prompt_name
# For scripts/scenes/sentences (documents):
doc_embeddings = model.encode(texts, prompt_name="document", normalize_embeddings=True)
# For search queries:
query_embedding = model.encode("a love confession", prompt_name="query", normalize_embeddings=True)
Requirements: pip install sentence-transformers peft

Important:

Always use prompt_name="document" for any content being indexed (scenes, sentences, scripts)
Always use prompt_name="query" for search queries
Always set normalize_embeddings=True — the model was trained with cosine similarity
Do not use SentenceTransformer("AnzeZ/fede-embeddinggemma") directly — it silently drops the LoRA weights and gives you base model quality