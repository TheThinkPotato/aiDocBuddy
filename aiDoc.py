# A script to load a PDF, chunk its text, store in ChromaDB, and answer questions using Ollama LLM.
# Make sure to have ollama, pypdf, chromadb installed and Ollama server

import ollama
import sys
from pypdf import PdfReader
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

# Setup your reasoning model, e.g., "llama3", "mistral", "deepseek-r1"
REASONING_MODEL = "deepseek-r1"
EMBEDDING_MODEL = "mxbai-embed-large"

# 1. Load PDF
if len(sys.argv) < 2:
    print("Error: No PDF file provided.")
    print("Usage: python aiDoc.py <path_to_pdf>")
    sys.exit(1)

input_pdf = sys.argv[1]

reader = PdfReader(input_pdf)
print(f"Extracting text from \"{input_pdf}\"...")
text = "\n".join([page.extract_text() for page in reader.pages])

print(f"Document has {len(text)} characters")

# 2. Split into chunks
chunk_size = 500
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# 3. Setup Chroma + embedding
client = PersistentClient(path="./chroma_db")
embedding_func = embedding_functions.OllamaEmbeddingFunction(model_name=EMBEDDING_MODEL)
collection = client.get_or_create_collection(name="pdf_docs", embedding_function=embedding_func)

# Add chunks to vector DB
for i, chunk in enumerate(chunks):
    collection.add(documents=[chunk], ids=[f"chunk_{i}"])

print("=========================================================================")
print("Setup complete. You can now ask questions about the document.")
while True:
    print("=========================================================================")
    question = input("Question: ")
    if(question.lower() in ['q', 'quit', 'exit']):
        print("Exiting...")
        sys.exit()
    if(question.strip() == ""):
        continue
    question = question.strip() + question.endswith("?") * "?"
    
    # 4. Query
    # question = "Name/List three (3) responsibilities of the MidTown IT management team?"
    results = collection.query(query_texts=[question], n_results=3)

    # 5. Send retrieved context + question to Ollama
    context = "\n".join(results["documents"][0])
    prompt = f"Answer the question based on the following context:\n{context}\n\nQuestion: {question}"
    response = ollama.chat(model=REASONING_MODEL, messages=[{"role": "user", "content": prompt}])

    print("\n-------------------------------------------------------------------------")
    print("Q:", question)    
    print("-------------------------------------------------------------------------\n")
    print(response["message"]["content"])
    key =  input("\n\nPress (Enter) to continue or (q) to quit...")
    if(key == 'q'):
        sys.exit()


