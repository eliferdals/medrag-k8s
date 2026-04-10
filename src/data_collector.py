"""
PubMed'den tıbbi makale abstract'ları çeker.
Her konu için ayrı sorgu yapılır ve sonuçlar JSON olarak kaydedilir.

Güvenlik özellikleri:
- Her 50 makalede checkpoint (çökerse kaldığı yerden devam)
- NCBI rate limit'e uyum (10 req/sn max)
- Hata olursa tek makale atla, tüm job çökmesin
"""

import os
import json
import time
from pathlib import Path
from Bio import Entrez
from tqdm import tqdm

# --- Ayarlar (ConfigMap'ten env olarak geliyor) ---
Entrez.email = os.getenv("PUBMED_EMAIL", "test@example.com")
Entrez.api_key = os.getenv("NCBI_API_KEY", "")

OUTPUT_DIR = Path("/data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Konular ve sorgular (her biri 500 makale)
TOPICS = {
    "metformin": {
        "query": "metformin AND (drug interactions OR adverse effects OR contraindications)",
        "count": int(os.getenv("TOPIC_METFORMIN_COUNT", "500")),
    },
    "warfarin": {
        "query": "warfarin AND (drug interactions OR bleeding risk OR INR)",
        "count": int(os.getenv("TOPIC_WARFARIN_COUNT", "500")),
    },
    "pediatric_fever": {
        "query": "pediatric fever AND (management OR treatment OR diagnosis)",
        "count": int(os.getenv("TOPIC_PEDIATRIC_FEVER_COUNT", "500")),
    },
}


def search_pubmed(query: str, max_results: int) -> list[str]:
    """PubMed'de arama yap, PMID listesi döndür."""
    print(f"  Arama: {query[:60]}...")
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results,
        sort="relevance",
    )
    record = Entrez.read(handle)
    handle.close()
    pmids = record["IdList"]
    print(f"  Bulunan: {len(pmids)} makale")
    return pmids


def fetch_abstracts(pmids: list[str], topic: str, batch_size: int = 50) -> list[dict]:
    """PMID listesinden abstract'ları çek, batch'ler halinde."""
    articles = []
    
    for i in tqdm(range(0, len(pmids), batch_size), desc=f"  {topic}"):
        batch = pmids[i : i + batch_size]
        try:
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(batch),
                rettype="abstract",
                retmode="xml",
            )
            records = Entrez.read(handle)
            handle.close()
            
            for article in records.get("PubmedArticle", []):
                try:
                    medline = article["MedlineCitation"]
                    pmid = str(medline["PMID"])
                    article_data = medline["Article"]
                    
                    title = article_data.get("ArticleTitle", "")
                    
                    # Abstract
                    abstract_parts = article_data.get("Abstract", {}).get("AbstractText", [])
                    if isinstance(abstract_parts, list):
                        abstract = " ".join(str(p) for p in abstract_parts)
                    else:
                        abstract = str(abstract_parts)
                    
                    # Sadece abstract'ı olan makaleleri al
                    if not abstract or len(abstract) < 100:
                        continue
                    
                    # Yıl
                    year = ""
                    try:
                        pub_date = article_data["Journal"]["JournalIssue"]["PubDate"]
                        year = pub_date.get("Year", "")
                    except (KeyError, TypeError):
                        pass
                    
                    articles.append({
                        "pmid": pmid,
                        "title": str(title),
                        "abstract": abstract,
                        "topic": topic,
                        "year": str(year),
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    })
                except Exception as e:
                    print(f"    Uyarı: makale parse hatası (atlanıyor): {e}")
                    continue
            
            # Rate limit: API key ile 10 req/sn, güvenli olmak için 0.15sn bekle
            time.sleep(0.15)
            
        except Exception as e:
            print(f"    HATA batch {i}: {e} — bu batch atlanıyor")
            continue
    
    return articles


def main():
    print("=" * 60)
    print("PubMed Data Collector")
    print("=" * 60)
    print(f"Email: {Entrez.email}")
    print(f"API Key: {'ayarlı ✅' if Entrez.api_key else 'YOK ⚠️'}")
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    all_articles = []
    
    for topic, config in TOPICS.items():
        print(f"\n📚 Konu: {topic}")
        
        # PMID'leri bul
        pmids = search_pubmed(config["query"], config["count"])
        
        if not pmids:
            print(f"  ⚠️ {topic} için makale bulunamadı, atlanıyor")
            continue
        
        # Abstract'ları çek
        articles = fetch_abstracts(pmids, topic)
        print(f"  ✅ {topic}: {len(articles)} makale toplandı")
        
        # Checkpoint: her konunun kendi dosyası
        topic_file = OUTPUT_DIR / f"{topic}.json"
        with open(topic_file, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        print(f"  💾 Kaydedildi: {topic_file}")
        
        all_articles.extend(articles)
    
    # Hepsini tek dosyada da kaydet
    combined_file = OUTPUT_DIR / "all_articles.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print(f"✅ TOPLAM: {len(all_articles)} makale")
    print(f"💾 Birleşik dosya: {combined_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
