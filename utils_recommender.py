import numpy as np
from sentence_transformers import SentenceTransformer

# Load lightweight embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Mock product database (In production, replace with ChromaDB/FAISS)
PRODUCT_DATABASE = [
    {"id": 1, "name": "Wireless Bluetooth Headphones", "category": "Electronics", "description": "Noise-cancelling over-ear headphones with 30h battery life."},
    {"id": 2, "name": "Smart Fitness Watch", "category": "Electronics", "description": "Waterproof smartwatch with heart rate monitor and GPS."},
    {"id": 3, "name": "Organic Green Tea", "category": "Food & Beverage", "description": "Premium loose-leaf green tea, rich in antioxidants."},
    {"id": 4, "name": "Eco-Friendly Yoga Mat", "category": "Sports", "description": "Non-slip yoga mat with carrying strap, made from recycled materials."},
    {"id": 5, "name": "Portable Phone Charger", "category": "Electronics", "description": "20000mAh fast-charging power bank for all devices."}
]

# Precompute embeddings
product_texts = [f"{p['name']} {p['category']} {p['description']}" for p in PRODUCT_DATABASE]
product_embeddings = model.encode(product_texts)

def get_recommendations(query_text, top_k=2):
    query_embedding = model.encode([query_text])
    similarities = np.dot(product_embeddings, query_embedding.T).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    recommendations = []
    for idx in top_indices:
        if similarities[idx] > 0.15: # Threshold to avoid irrelevant results
            recommendations.append(PRODUCT_DATABASE[idx])
    return recommendations