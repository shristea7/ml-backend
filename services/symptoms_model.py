import json
from sentence_transformers import SentenceTransformer, util

# Load medicine dataset
with open("data/medicines_1000.json") as f:
    medicines = json.load(f)

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Precompute medicine uses embeddings
medicine_uses = [", ".join(med.get("uses", [])) for med in medicines]
medicine_embeddings = model.encode(medicine_uses, convert_to_tensor=True)


def detect_symptoms(user_input, top_k=7):
    """
    Detects top medicines for given symptom or user text.
    Returns top_k medicines (id, name, brand).
    """
    query_emb = model.encode(user_input, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, medicine_embeddings)[0]
    top_indices = scores.topk(top_k).indices.tolist()

    result = []
    for idx in top_indices:
        med = medicines[idx]
        result.append({
            "id": med["id"],
            "name": med["name"],
            "brand": med.get("brand", "")
        })
    return result
