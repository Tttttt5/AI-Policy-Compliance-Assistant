from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# -------------------------------
# Step 1️ — Load local embeddings & Chroma vector database
# -------------------------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="chroma_store", embedding_function=embeddings)

# -------------------------------
# Step 2️ — Advanced Retriever for diverse & relevant chunks
# -------------------------------
retriever = db.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": 10, "fetch_k": 30, "lambda_mult": 0.7}
)

# -------------------------------
# Step 3️ — Your question
# -------------------------------
question = "Compare GDPR and India's DPDP Act in terms of user consent and data retention."

# -------------------------------
# Step 4️ — Retrieve relevant chunks
# -------------------------------
docs = retriever.get_relevant_documents(question)

# Make sure chunks are from more than one document
sources = {d.metadata.get("source", "unknown") for d in docs}
if len(sources) < 2:
    print(" Retrieved chunks are mostly from one document. Expanding search...")
    more_docs = db.similarity_search(question, k=5)
    docs.extend(more_docs)
    sources = {d.metadata.get("source", "unknown") for d in docs}

# -------------------------------
# Step 5️ — Preview retrieved chunks (for debugging)
# -------------------------------
print(f"\n Retrieved {len(docs)} chunks for question:\n{question}\n")
for i, d in enumerate(docs[:5]):
    print(f"--- Chunk {i+1} (from {d.metadata.get('source', 'unknown')}) ---")
    print(d.page_content[:400], "\n")

# -------------------------------
# Step 6️ — Connect to your local LLM (Ollama)
# -------------------------------
#  Tip: Use a small model like gemma:2b or phi3:mini for 8GB RAM systems
llm = Ollama(model="gemma:2b", temperature=0.2)

# -------------------------------
# Step 7️ — Build a retrieval + reasoning chain
# -------------------------------
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# -------------------------------
# Step 8️ — Build a smart comparison prompt
# -------------------------------
context_text = "\n\n".join(
    [f"From {d.metadata.get('source', 'unknown')}:\n{d.page_content[:800]}" for d in docs[:6]]
)

compare_prompt = (
    "You are an expert legal analyst specializing in data protection and AI ethics. "
    "You have been given excerpts from two legal documents: "
    "the European Union's General Data Protection Regulation (GDPR) and "
    "India's Digital Personal Data Protection Act (DPDP Act, 2023). "
    "Use ONLY the provided context below to produce a concise comparison focusing on "
    "user consent, data retention, and key differences. "
    "If one law does not specify something, explicitly mention that.\n\n"
    f"Context:\n{context_text}\n\n"
    f"Question: {question}"
)

# -------------------------------
# Step 9️ — Run the comparison
# -------------------------------
print("\n Generating AI answer, please wait...\n")
answer = qa.run(compare_prompt)

# -------------------------------
# Step 10 — Show final result
# -------------------------------
print("❓ Question:", question)
print("💡 Answer:", answer)

print("\n📚 Sources used:", ", ".join(list(sources)))
 
