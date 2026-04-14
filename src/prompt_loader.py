"""Prompt yükleyici — versiyonlanmış YAML'dan okur.

Faz 5'te prompts/ klasörü ConfigMap olarak /app/prompts'a mount edilir.
Yeni versiyon eklemek için:
    1) prompts/rag_prompt_v3.yaml dosyası ekle
    2) kubectl create configmap rag-prompts -n medrag \\
         --from-file=prompts/ --dry-run=client -o yaml | kubectl apply -f -
    3) ConfigMap PROMPT_VERSION değerini v3 yap, pod'u restart et:
         kubectl rollout restart deploy/rag-api -n medrag

Kullanım (rag_chain.py içinden):
    from prompt_loader import load_prompt
    p = load_prompt(os.getenv("PROMPT_VERSION", "v1"))
    system, user = p["system"], p["user_template"]
"""
import os
from pathlib import Path
import yaml

# Container içinde ConfigMap mount yolu. Override edilebilir.
PROMPTS_DIR = Path(os.getenv("PROMPTS_DIR", "/app/prompts"))


def load_prompt(version: str = "v1") -> dict:
    path = PROMPTS_DIR / f"rag_prompt_{version}.yaml"
    if not path.exists():
        available = ", ".join(list_versions()) or "(boş)"
        raise FileNotFoundError(
            f"Prompt version '{version}' yok. Mevcut: {available}"
        )
    with open(path) as f:
        return yaml.safe_load(f)


def list_versions() -> list[str]:
    if not PROMPTS_DIR.exists():
        return []
    return sorted(
        p.stem.replace("rag_prompt_", "")
        for p in PROMPTS_DIR.glob("rag_prompt_*.yaml")
    )
