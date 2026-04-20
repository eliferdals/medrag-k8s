# MedRAG — Kubernetes Üzerinde Tıbbi RAG Sistemi

![mimarik8s](./images/mimarik8s.png)

PubMed makalelerini bilgi kaynağı olarak kullanan, tıbbi soruları yanıtlayan bir RAG (Retrieval-Augmented Generation) sistemi. Tüm bileşenler Kubernetes üzerinde çalışır.

## Proje Hakkında

MedRAG, üç tıbbi alanda PubMed'den toplanmış 1500 makale abstract'ını kullanarak kullanıcı sorularına kaynak göstererek cevap veren bir sistemdir:

- **Metformin** — yan etkiler, ilaç etkileşimleri, kontrendikasyonlar
- **Warfarin** — kanama riski, ilaç etkileşimleri, INR yönetimi
- **Pediatric Fever Management** — çocuklarda ateş yönetimi

## Tech Stack

- **Orchestration:** Kubernetes (Docker Desktop)
- **RAG Framework:** LangChain
- **Vector Database:** ChromaDB
- **LLM:** Ollama (Phi-3 mini, lokal)
- **Embedding:** HuggingFace `all-MiniLM-L6-v2`
- **API:** FastAPI
- **UI:** Streamlit
- **Database:** PostgreSQL (CNPG operator) — sorgu loglama ve analitik
- **Monitoring:** Prometheus + Grafana
- **Packaging:** Helm

## Mimari
Kullanıcı → Ingress (medrag.local)
├─ / → Streamlit UI
└─ /api → FastAPI RAG Servisi
├─ ChromaDB (vector search)
├─ Ollama (LLM)
└─ PostgreSQL (query logs)
## Kurulum

(Kurulum detayları daha sonra eklenecek)

## Lisans

MIT
