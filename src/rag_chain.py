"""RAG chain: classifier guardrail -> ChromaDB retrieval -> Ollama LLM."""
import os, pickle, time
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from prompts import SYSTEM_PROMPT, RAG_TEMPLATE, format_context, extract_sources

CHROMA_DIR = os.getenv("CHROMA_DIR", "/data/chroma")
CLASSIFIER_PATH = os.getenv("CLASSIFIER_PATH", "/data/classifier/clf.pkl")
CLASSIFIER_THRESHOLD = float(os.getenv("CLASSIFIER_THRESHOLD", "0.70"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama.medrag.svc.cluster.local:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
TOP_K = int(os.getenv("TOP_K", "4"))

class RAGPipeline:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        self.vs = Chroma(persist_directory=CHROMA_DIR, embedding_function=self.embeddings, collection_name=os.getenv("COLLECTION_NAME", "medrag"))
        self.retriever = self.vs.as_retriever(search_kwargs={"k": TOP_K})
        self.llm = OllamaLLM(base_url=OLLAMA_URL, model=OLLAMA_MODEL, temperature=0.1)
        with open(CLASSIFIER_PATH, "rb") as f:
            self.clf = pickle.load(f)

    def classify(self, q):
        proba = self.clf.predict_proba([q])[0]
        med_idx = list(self.clf.classes_).index("medical") if "medical" in self.clf.classes_ else 1
        score = float(proba[med_idx])
        label = "medical" if score >= CLASSIFIER_THRESHOLD else "off_topic"
        return label, score

    def answer(self, question):
        t0 = time.time()
        label, score = self.classify(question)
        if label == "off_topic":
            return {"answer": "Bu soru medikal literatür kapsamı dışında görünüyor. Lütfen PubMed tabanlı tıbbi bir soru sorun.",
                    "sources": [], "classifier_score": score, "classifier_label": label,
                    "latency_ms": int((time.time()-t0)*1000), "model": OLLAMA_MODEL}
        docs = self.retriever.invoke(question)
        context = format_context(docs)
        prompt = f"{SYSTEM_PROMPT}\n\n{RAG_TEMPLATE.format(context=context, question=question)}"
        answer_text = self.llm.invoke(prompt)
        return {"answer": answer_text, "sources": extract_sources(docs),
                "classifier_score": score, "classifier_label": label,
                "latency_ms": int((time.time()-t0)*1000), "model": OLLAMA_MODEL}

_pipeline = None
def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
