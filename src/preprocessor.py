"""
Abstract'ları chunk'lara böler ve metadata ekler.
LangChain RecursiveCharacterTextSplitter kullanır.
"""

import json
import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

INPUT_FILE = Path("/data/raw/all_articles.json")
OUTPUT_FILE = Path("/data/chunks/chunks.json")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))


def main():
    print("=" * 60)
    print("Text Preprocessor / Chunker")
    print("=" * 60)
    print(f"Input: {INPUT_FILE}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Overlap: {CHUNK_OVERLAP}")
    print()
    
    if not INPUT_FILE.exists():
        print(f"❌ HATA: {INPUT_FILE} bulunamadı!")
        print("   Önce data_collector.py çalıştırılmalı.")
        return
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        articles = json.load(f)
    
    print(f"📖 Yüklenen makale sayısı: {len(articles)}")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    all_chunks = []
    
    for article in articles:
        # Title + Abstract'ı birleştir
        text = f"{article['title']}\n\n{article['abstract']}"
        
        chunks = splitter.split_text(text)
        
        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{article['pmid']}_{i}",
                "text": chunk_text,
                "metadata": {
                    "pmid": article["pmid"],
                    "title": article["title"],
                    "topic": article["topic"],
                    "year": article["year"],
                    "url": article["url"],
                    "chunk_index": i,
                },
            })
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Toplam chunk: {len(all_chunks)}")
    print(f"💾 Kaydedildi: {OUTPUT_FILE}")
    
    # Konu dağılımı
    topic_counts = {}
    for chunk in all_chunks:
        topic = chunk["metadata"]["topic"]
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    print("\n📊 Konu dağılımı:")
    for topic, count in topic_counts.items():
        print(f"  {topic}: {count} chunk")


if __name__ == "__main__":
    main()
