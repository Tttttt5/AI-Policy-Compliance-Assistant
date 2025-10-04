import os
import re
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Step 1  — Folder where your documents live
data_folder = "data"

# Step 2 — Load all text and PDF documents
docs = []
for file in os.listdir(data_folder):
    path = os.path.join(data_folder, file)
    if file.endswith(".pdf"):
        print(f" Loading PDF: {file}")
        loader = PyPDFLoader(path)
        file_docs = loader.load()
    elif file.endswith(".txt"):
        print(f" Loading Text: {file}")
        loader = TextLoader(path, encoding="utf-8")
        file_docs = loader.load()
    else:
        continue

    # Add metadata to know which file each chunk came from
    for d in file_docs:
        d.metadata["source"] = file

    docs.extend(file_docs)

# Step 3 — Clean text to remove broken words or extra spaces
def clean_text(text):
    text = re.sub(r"-\s*\n", "", text)      # joins hyphenated line breaks
    text = re.sub(r"\n+", " ", text)        # replaces newlines with spaces
    text = re.sub(r"\s{2,}", " ", text)     # removes double spaces
    return text.strip()

for d in docs:
    d.page_content = clean_text(d.page_content)

# Step 4 — Split text intelligently into meaningful chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=250,
    separators=["\n\n", "\n", ".", "!", "?", " "]
)
split_docs = splitter.split_documents(docs)
print(f" Split into {len(split_docs)} clean chunks")

# Step 5️ — Create vector embeddings (numerical meaning of text)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Step 6️ — Save all vectors in a local database (Chroma)
db = Chroma.from_documents(split_docs, embedding=embeddings, persist_directory="chroma_store")
db.persist()

print(" All documents processed, cleaned, and saved to Chroma!")
