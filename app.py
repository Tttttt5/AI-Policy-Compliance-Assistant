import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# -------------------------------
# 🌟 PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AI Policy Compliance Assistant", page_icon="🤖", layout="wide")

st.title("🤖 AI Policy Compliance Assistant")
st.write("Ask questions about GDPR, India's DPDP Act, and AI Ethics — all running locally with Ollama.")

# -------------------------------
# ⚙️ LOAD SYSTEM COMPONENTS
# -------------------------------
@st.cache_resource
def load_system():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="chroma_store", embedding_function=embeddings)
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 30, "lambda_mult": 0.7}
    )
    # 👇 Choose a model that fits your memory
    llm = Ollama(model="phi3:mini", temperature=0.3)  # tinyllama / gemma:2b / mistral
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa, retriever

qa, retriever = load_system()

# -------------------------------
# 🧩 SIDEBAR
# -------------------------------
st.sidebar.header("📄 Documents Loaded:")
st.sidebar.markdown("""
- 🇪🇺 GDPR (EU Data Protection)
- 🇮🇳 India DPDP Act (2023)
- 🌍 UNESCO AI Ethics Guidelines
""")
st.sidebar.info("All data is processed locally — no internet or API keys required ✅")

# -------------------------------
# 💬 QUESTION INPUT
# -------------------------------
st.markdown("---")
st.subheader("💬 Ask a Question or Compare Laws")

question = st.text_area(
    "Enter your question below:",
    placeholder="e.g., Compare GDPR and India's DPDP Act on user consent and data retention.",
    height=100
)

# -------------------------------
# 🚀 ANALYSIS LOGIC
# -------------------------------
if st.button("Generate Answer 🚀"):
    if question.strip() == "":
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Retrieving relevant context..."):
            docs = retriever.get_relevant_documents(question)

        sources = {d.metadata.get("source", "unknown") for d in docs}

        # Show retrieved context
        with st.expander("🔍 Retrieved Context (Top 5 Chunks)"):
            for i, d in enumerate(docs[:5]):
                st.markdown(f"**Chunk {i+1} — from _{d.metadata.get('source', 'unknown')}_**")
                st.write(d.page_content[:600] + "...")
                st.markdown("---")

        # Build structured comparison prompt
        context_text = "\n\n".join(
            [f"From {d.metadata.get('source', 'unknown')}:\n{d.page_content[:800]}" for d in docs[:6]]
        )

        compare_prompt = f"""
You are a senior legal policy analyst specializing in data protection and AI ethics.
Use only the provided context from GDPR, DPDP, and AI Ethics guidelines to answer the user's question.

Present the answer in a structured, professional format using Markdown headings and bullet points.

📚 Context from documents:
{context_text}

❓ Question:
{question}

🧾 Output Format:
### 🧭 Overview
### 🇪🇺 GDPR Highlights
### 🇮🇳 DPDP Act Highlights
### ⚖️ Key Similarities
### 🚩 Key Differences
### 🧩 Summary
"""

        with st.spinner("🤖 Generating structured answer..."):
            answer = qa.run(compare_prompt)

        st.subheader("📊 Final Answer")
        st.markdown(answer)
        st.markdown(f"📚 **Sources used:** {', '.join(list(sources))}")
        st.success("✅ Done!")