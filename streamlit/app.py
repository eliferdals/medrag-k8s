"""MedRAG Streamlit UI — JWT auth + query + history."""
import os
import requests
import streamlit as st

# Service iç adı: rag-api Service port 80'de açık (targetPort 8000).
# Cluster içinden Streamlit → rag-api çağrısı bu yüzden :80 kullanır.
API = os.getenv("RAG_API_URL", "http://rag-api.medrag.svc.cluster.local:80")

st.set_page_config(page_title="MedRAG", page_icon="🩺", layout="wide")

if "token" not in st.session_state:
    st.session_state.token = None
    st.session_state.username = None


def auth_headers():
    return {"Authorization": f"Bearer {st.session_state.token}"}


def login_view():
    st.title("🩺 MedRAG")
    st.caption("PubMed tabanlı tıbbi literatür Q&A")
    tab_login, tab_reg = st.tabs(["Giriş", "Kayıt"])
    with tab_login:
        u = st.text_input("Kullanıcı adı", key="lu")
        p = st.text_input("Şifre", type="password", key="lp")
        if st.button("Giriş yap", type="primary"):
            try:
                r = requests.post(
                    f"{API}/login",
                    data={"username": u, "password": p},
                    timeout=10,
                )
                if r.status_code == 200:
                    st.session_state.token = r.json()["access_token"]
                    st.session_state.username = u
                    st.rerun()
                else:
                    st.error(f"Giriş başarısız ({r.status_code})")
            except requests.RequestException as e:
                st.error(f"API'ye ulaşılamadı: {e}")
    with tab_reg:
        u2 = st.text_input("Kullanıcı adı", key="ru")
        p2 = st.text_input("Şifre (en az 6 karakter)", type="password", key="rp")
        if st.button("Kayıt ol"):
            try:
                r = requests.post(
                    f"{API}/register",
                    json={"username": u2, "password": p2},
                    timeout=10,
                )
                if r.status_code == 200:
                    st.success("Kayıt başarılı, Giriş sekmesinden devam et")
                else:
                    st.error(f"Kayıt başarısız: {r.text}")
            except requests.RequestException as e:
                st.error(f"API'ye ulaşılamadı: {e}")


def query_view():
    st.header("Tıbbi Literatür Sorgusu")
    q = st.text_area(
        "Sorunuz (İngilizce daha iyi sonuç verir):",
        height=100,
        placeholder="e.g., What is the first-line treatment for type 2 diabetes?",
    )
    if st.button("Sor", type="primary") and q.strip():
        with st.spinner("Yanıt oluşturuluyor… (ilk soru ~60-90sn sürebilir)"):
            try:
                r = requests.post(
                    f"{API}/query",
                    json={"question": q},
                    headers=auth_headers(),
                    timeout=180,
                )
            except requests.RequestException as e:
                st.error(f"API hatası: {e}")
                return

        if r.status_code == 200:
            d = r.json()
            st.markdown("### Yanıt")
            st.write(d["answer"])

            if d.get("sources"):
                st.markdown("### Kaynaklar")
                for s in d["sources"]:
                    pmid = s.get("pmid", "?")
                    title = s.get("title", "(başlık yok)")
                    st.markdown(
                        f"- **PMID [{pmid}]"
                        f"(https://pubmed.ncbi.nlm.nih.gov/{pmid}/)** — {title}"
                    )

            c1, c2, c3 = st.columns(3)
            c1.metric("Latency", f"{d.get('latency_ms', 0)} ms")
            score = d.get("classifier_score")
            c2.metric("Classifier", f"{score:.2f}" if score is not None else "-")
            c3.metric("Model", d.get("model", "-"))
        elif r.status_code == 401:
            st.error("Oturumun süresi doldu, tekrar giriş yap.")
            st.session_state.token = None
        elif r.status_code == 422:
            st.warning(
                "Bu soru tıbbi içerikli görünmüyor "
                "(classifier guardrail). Lütfen tıbbi bir soru sor."
            )
        else:
            st.error(f"Hata {r.status_code}: {r.text}")


def history_view():
    st.header("Sorgu Geçmişi")
    try:
        r = requests.get(
            f"{API}/history?limit=20", headers=auth_headers(), timeout=10
        )
    except requests.RequestException as e:
        st.error(f"API hatası: {e}")
        return

    if r.status_code != 200:
        st.error(f"Geçmiş alınamadı ({r.status_code})")
        return

    items = r.json().get("items", [])
    if not items:
        st.info("Henüz sorgu yok.")
        return

    for row in items:
        title = row["question"][:80] + ("…" if len(row["question"]) > 80 else "")
        with st.expander(f"🕐 {row['created_at']} — {title}"):
            st.markdown("**Soru:**")
            st.write(row["question"])
            st.markdown("**Yanıt:**")
            st.write(row["answer"])
            score = row.get("score")
            lat = row.get("latency_ms", 0)
            st.caption(
                f"Score: {score:.2f if score is not None else '-'} · {lat} ms"
            )


def main_view():
    with st.sidebar:
        st.markdown(f"**👤 {st.session_state.username}**")
        if st.button("Çıkış"):
            st.session_state.token = None
            st.session_state.username = None
            st.rerun()
        page = st.radio("Sayfa", ["Sorgu", "Geçmiş"])

    if page == "Sorgu":
        query_view()
    else:
        history_view()


(main_view if st.session_state.token else login_view)()

st.markdown("---")
st.caption(
    "*MedRAG — Research tool based on PubMed literature. "
    "Not a substitute for professional medical consultation.*"
)
