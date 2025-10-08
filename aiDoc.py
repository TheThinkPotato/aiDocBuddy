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

# 1. Load PDFs
if len(sys.argv) < 2:
    print("Error: No PDF file(s) provided.")
    print("Usage: python aiDoc.py <path_to_pdf1> [path_to_pdf2] [path_to_pdf3] ...")
    sys.exit(1)

input_pdfs = sys.argv[1:]
all_documents = []

for pdf_path in input_pdfs:
    try:
        reader = PdfReader(pdf_path)
        print(f"Extracting text from \"{pdf_path}\"...")
        text = "\n".join([page.extract_text() for page in reader.pages])
        all_documents.append({
            'path': pdf_path,
            'text': text,
            'filename': pdf_path.split('/')[-1].split('\\')[-1]  # Get filename from path
        })
        print(f"Document \"{pdf_path}\" has {len(text)} characters")
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        continue

if not all_documents:
    print("Error: No valid PDF documents could be loaded.")
    sys.exit(1)

total_chars = sum(len(doc['text']) for doc in all_documents)
print(f"Total characters across {len(all_documents)} document(s): {total_chars}")

# 2. Split into chunks with document source tracking
chunk_size = 500
all_chunks = []
chunk_metadata = []

for doc_idx, document in enumerate(all_documents):
    text = document['text']
    filename = document['filename']
    
    # Create chunks for this document
    doc_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    # Add chunks with metadata
    for chunk_idx, chunk in enumerate(doc_chunks):
        all_chunks.append(chunk)
        chunk_metadata.append({
            'source_file': filename,
            'doc_index': doc_idx,
            'chunk_index': chunk_idx
        })

print(f"Created {len(all_chunks)} chunks from {len(all_documents)} document(s)")

# 3. Setup Chroma + embedding
client = PersistentClient(path="./chroma_db")
embedding_func = embedding_functions.OllamaEmbeddingFunction(model_name=EMBEDDING_MODEL)
collection = client.get_or_create_collection(name="pdf_docs", embedding_function=embedding_func)

# Add chunks to vector DB with metadata
for i, (chunk, metadata) in enumerate(zip(all_chunks, chunk_metadata)):
    collection.add(
        documents=[chunk], 
        ids=[f"chunk_{i}"],
        metadatas=[metadata]
    )

print("=========================================================================")
print(f"Setup complete. You can now ask questions about the {len(all_documents)} document(s).")
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
    results = collection.query(query_texts=[question], n_results=3)

    # 5. Send retrieved context + question to Ollama
    context_parts = []
    source_files = set()
    
    for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        if metadata is not None:
            source_file = metadata.get('source_file', 'Unknown')
        else:
            source_file = 'Unknown'
        source_files.add(source_file)
        context_parts.append(f"[From {source_file}]: {doc}")
    
    context = "\n\n".join(context_parts)
    prompt = f"Answer the question based on the following context from multiple documents:\n{context}\n\nQuestion: {question}"
    response = ollama.chat(model=REASONING_MODEL, messages=[{"role": "user", "content": prompt}])

    print("\n-------------------------------------------------------------------------")
    print("Q:", question)
    print(f"Sources: {', '.join(sorted(source_files))}")
    print("-------------------------------------------------------------------------\n")
    print(response["message"]["content"])
    key =  input("\n\nPress (Enter) to continue or (q) to quit...")
    if(key == 'q'):
        sys.exit()


