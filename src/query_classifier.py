"""
Basit bir tıbbi soru sınıflandırıcısı eğitir.
TF-IDF + Logistic Regression. .pkl olarak PVC'ye kaydeder.
"""
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from training_data import get_training_data

OUTPUT_DIR = Path("/data/models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUTPUT_DIR / "query_classifier.pkl"


def main():
    print("=" * 60)
    print("Query Classifier Training")
    print("=" * 60)
    
    texts, labels = get_training_data()
    print(f"📖 Toplam örnek: {len(texts)}")
    print(f"  Tıbbi: {sum(labels)}")
    print(f"  Tıbbi değil: {len(labels) - sum(labels)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"\n📊 Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Pipeline: TF-IDF + Logistic Regression
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=1000,
            lowercase=True,
            stop_words="english",
        )),
        ("clf", LogisticRegression(max_iter=1000, C=1.0)),
    ])
    
    print("\n⚙️  Eğitim başlıyor...")
    pipeline.fit(X_train, y_train)
    print("  ✅ Eğitim tamamlandı")
    
    # Değerlendirme
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n📈 Test accuracy: {acc:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Non-medical", "Medical"]))
    
    # Test örnekleri
    print("🧪 Test tahminleri:")
    test_queries = [
        "What are the interactions between warfarin and aspirin?",
        "How do I bake a chocolate cake?",
        "Is metformin safe in pregnancy?",
        "What's the weather like?",
        "Child fever 39 degrees emergency?",
    ]
    for q in test_queries:
        pred = pipeline.predict([q])[0]
        prob = pipeline.predict_proba([q])[0][1]
        label = "TIBBİ ✅" if pred == 1 else "TIBBİ DEĞİL ❌"
        print(f"  [{prob:.2%}] {label}: {q}")
    
    # Kaydet
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\n💾 Model kaydedildi: {MODEL_PATH}")
    print(f"   Boyut: {MODEL_PATH.stat().st_size / 1024:.1f} KB")
    
    print("\n" + "=" * 60)
    print("✅ FAZ 3 TAMAMLANDI")
    print("=" * 60)


if __name__ == "__main__":
    main()
