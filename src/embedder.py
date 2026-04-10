"""
Chunk'ları embed eder ve ChromaDB'ye yükler.
HuggingFace sentence-transformers ile LOKAL embedding üretir.
"""

import json
import os
from pathlib import Path
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

INPUT_FILE = Path("/data/chunks/chunks.json")
CHROMA_DIR = Path("/data/chroma_db")
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = "medrag"
BATCH_SIZE = 100


def main():
    print("=" * 60)
    print("Embedder + ChromaDB Loader")
    print("=" * 60)
    print(f"Input: {INPUT_FILE}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"ChromaDB: {CHROMA_DIR}")
    print()
    
    if not INPUT_FILE.exists():
        print(f"❌ HATA: {INPUT_FILE} bulunamadı!")
        return
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print(f"📖 Yüklenen chunk sayısı: {len(chunks)}")
    
    # Embedding modeli yükle (ilk seferde indirir, sonra cache'den)
    print(f"\n🤗 Embedding modeli yükleniyor: {EMBEDDING_MODEL}")
    print("  (İlk seferde ~90 MB indirilir, bu NORMAL)")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("  ✅ Model hazır")
    
    # ChromaDB client
    print(f"\n📦 ChromaDB bağlantısı...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    
    # Eski collection varsa sil (temiz başlangıç)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Eski collection silindi")
    except Exception:
        pass
    
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"  ✅ Collection oluşturuldu: {COLLECTION_NAME}")
    
    # Batch'ler halinde ekle
    print(f"\n⚙️  Embedding ve yükleme (batch size: {BATCH_SIZE})")
    
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="  Batches"):
        batch = chunks[i : i + BATCH_SIZE]
        
        ids = [c["chunk_id"] for c in batch]
        texts = [c["text"] for c in batch]
        metadatas = [c["metadata"] for c in batch]
        
        # Embed
        vectors = embeddings.embed_documents(texts)
        
        # ChromaDB'ye ekle
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=vectors,
            metadatas=metadatas,
        )
    
    # Kontrol
    count = collection.count()
    print(f"\n✅ ChromaDB'ye yüklenen chunk sayısı: {count}")
    
    # Test sorgusu (LLM yok, sadece retrieval test)
    print("\n🧪 Test retrieval sorgusu:")
    test_query = "What are the side effects of metformin?"
    print(f"  Soru: {test_query}")
    
    query_vector = embeddings.embed_query(test_query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=3,
    )
    
    print(f"  İlk 3 sonuç:")
    for idx, (doc_id, doc, meta) in enumerate(zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
    )):
        print(f"    {idx+1}. PMID {meta['pmid']} ({meta['topic']})")
        print(f"       {doc[:100]}...")
    
    print("\n" + "=" * 60)
    print(f"✅ FAZ 2 TAMAMLANDI: {count} chunk ChromaDB'de hazır")
    print("=" * 60)


if __name__ == "__main__":
    main()
