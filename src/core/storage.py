from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

client = QdrantClient(path="data/qdrant")    # Using local persistent storage

def store_vectors(name: str, chunks, embeddings):
    if not client.collection_exists(name):
        client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE)
        )
    client.upsert(
        collection_name=name,
        points=[
            PointStruct(id=i, vector=embeddings[i].tolist(), payload={"text": chunks[i]})
            for i in range(len(chunks))
        ]
    )
    

def fetch_emb(name: str, query_emb, limit: int):
    hits = client.search(name, query_emb.tolist(), limit=limit)
    return [h.payload["text"] for h in hits]