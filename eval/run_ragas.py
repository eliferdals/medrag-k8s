"""RAGAS ile offline RAG değerlendirme.

Kullanım:
    # 1) port-forward (kolay yol)
    kubectl port-forward -n medrag svc/rag-api 8080:80

    # 2) login → JWT
    TOKEN=$(curl -s -X POST http://localhost:8080/login \\
      -d "username=elif&password=YOUR_PASS" | jq -r .access_token)

    # 3) eval (LLM-as-judge için OPENAI_API_KEY gerekir; lokal LLM ile
    #    çalıştırmak istersen --judge ollama bayrağıyla aşağıdaki kodu uyarla)
    export RAG_API_URL=http://localhost:8080
    export RAG_TOKEN=$TOKEN
    export OPENAI_API_KEY=sk-...
    python eval/run_ragas.py --dataset eval/testset.jsonl --prompt-version v1

Çıktı: eval/results/<timestamp>_<version>.json + ekrana özet tablo.

Metrikler:
  - faithfulness        → cevap context'e ne kadar sadık (hallucination tespiti)
  - answer_relevancy    → cevap soruyla ne kadar alakalı
  - context_precision   → retrieve edilen context'in ne kadarı işe yaradı
  - context_recall      → ground_truth bilgisi context'te var mıydı

Kurulum:
    pip install ragas==0.2.6 datasets pandas openai
"""
import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from datasets import Dataset

API = os.getenv("RAG_API_URL", "http://localhost:8080")
TOKEN = os.getenv("RAG_TOKEN", "")


def run_queries(testset_path: Path) -> list[dict]:
    """Testset JSONL: {question, ground_truth, ground_truth_pmids}

    Not: rag-api /query yanıtı sources içinde sadece pmid+title döner.
    Gerçek chunk içeriği için /query yanıtına 'context_chunks' alanı
    eklemen lazım (Faz 4 main.py'de küçük bir değişiklik). Eğer henüz
    eklenmediyse, fallback olarak title kullanırız (context_recall düşük çıkar).
    """
    rows = []
    for line in testset_path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)

        r = requests.post(
            f"{API}/query",
            json={"question": item["question"]},
            headers={"Authorization": f"Bearer {TOKEN}"},
            timeout=180,
        )
        r.raise_for_status()
        d = r.json()

        # Tercih: gerçek chunk text'leri; yoksa fallback olarak title+pmid
        if "context_chunks" in d and d["context_chunks"]:
            contexts = d["context_chunks"]
        else:
            contexts = [
                f"[PMID {s.get('pmid', '?')}] {s.get('title', '')}"
                for s in d.get("sources", [])
            ]

        rows.append({
            "question": item["question"],
            "answer": d["answer"],
            "contexts": contexts,
            "reference": item["ground_truth"],   # RAGAS 0.2+ ismi
        })
        print(f"  ✓ {item['question'][:60]}…")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--prompt-version", default="v1")
    args = ap.parse_args()

    if not TOKEN:
        raise SystemExit("RAG_TOKEN ortam değişkeni boş — login yapıp set et.")

    print(f"→ Testset yükleniyor: {args.dataset}")
    rows = run_queries(Path(args.dataset))
    print(f"→ {len(rows)} soru tamamlandı, RAGAS başlıyor…")

    # Lazy import — RAGAS ağır
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness, answer_relevancy,
        context_precision, context_recall,
    )

    ds = Dataset.from_list(rows)
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy,
                 context_precision, context_recall],
    )

    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{int(time.time())}_{args.prompt_version}.json"

    df = result.to_pandas()
    out.write_text(df.to_json(orient="records", indent=2))

    print("\n=== ÖZET ===")
    print(df[["faithfulness", "answer_relevancy",
              "context_precision", "context_recall"]].mean())
    print(f"\nDetay → {out}")


if __name__ == "__main__":
    main()
