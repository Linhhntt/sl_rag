import os
import json
import time
import shutil
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

def load_jsonl(file_path):
    print(f"📂 Loading documents from {file_path}...")
    documents = []
    
    # First, count total lines
    total_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="  Loading", unit="doc"):
            try:
                data = json.loads(line)
                # Create a LangChain Document
                doc = Document(
                    page_content=data['text'],
                    metadata={
                        "docid": data['docid'],
                        "url": data['url'],
                        "credibility": data['cred'],  # This can be 'a' in Subjective Logic
                        "timestamp": data['timestamp']
                    }
                )
                documents.append(doc)
            except json.JSONDecodeError as e:
                print(f"  ⚠️  Error parsing JSON: {e}")
                continue
    
    print(f"✅ Loaded {len(documents)} total documents")
    return documents

def run_ingestion():
    start_time = time.time()
    print("🚀 Starting document ingestion process...")
    
    # 1. Load from your specific JSONL file
    jsonl_path = '../dataset/extracted_docs.jsonl'
    print(f"\n📂 Step 1: Loading JSONL file...")
    all_docs = load_jsonl(jsonl_path)
    
    # 2. Setup embeddings early to check existing DB
    print(f"\n🧠 Step 2: Setting up embeddings...")
    embedding_model = os.getenv("EMBEDDING_MODEL")
    print(f"  Using embedding model: {embedding_model}")
    embeddings = OllamaEmbeddings(model=embedding_model)
    
    # 3. Check existing database and get already processed documents
    print(f"\n💾 Step 3: Checking existing Chroma database...")
    chroma_path = "./chroma_db"
    print(f"  Database path: {chroma_path}")
    
    processed_docids = set()
    db_exists = os.path.exists(chroma_path)
    
    if db_exists:
        print("  📊 Existing database found, checking processed documents...")
        try:
            vector_db = Chroma(
                persist_directory=chroma_path,
                embedding_function=embeddings,
                collection_name="documents"
            )
            existing_data = vector_db.get()
            if existing_data['metadatas']:
                processed_docids = {meta['docid'] for meta in existing_data['metadatas']}
                print(f"  ✅ Found {len(processed_docids)} already processed documents")
        except Exception as e:
            print(f"  ⚠️  Could not read existing database: {e}")
            print("  Will start fresh...")
            shutil.rmtree(chroma_path)
            db_exists = False
    else:
        print("  🆕 No existing database found, starting fresh...")
    
    # 4. Filter documents to only new ones
    print(f"\n✂️  Step 4: Filtering and chunking documents...")
    new_docs = [doc for doc in tqdm(all_docs, desc="  Filtering", unit="doc") 
                if doc.metadata['docid'] not in processed_docids]
    print(f"  📊 Total documents: {len(all_docs)}")
    print(f"  📊 Already processed: {len(processed_docids)}")
    print(f"  📊 New documents to process: {len(new_docs)}")
    
    if not new_docs:
        print("  ℹ️  No new documents to process. Exiting.")
        return 0
    
    # Chunk only new documents
    print(f"  Using chunk_size=600, chunk_overlap=100")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    
    # Split with progress bar
    chunks = []
    for doc in tqdm(new_docs, desc="  Chunking", unit="doc"):
        chunks.extend(text_splitter.split_documents([doc]))
    
    print(f"  ✅ Created {len(chunks)} chunks from {len(new_docs)} new documents")
    
    # 5. Connect or create database
    print(f"\n💾 Step 5: Processing chunks into Chroma database...")
    
    if not db_exists:
        print("  🆕 Creating fresh database...")
        print(f"  ➕ Embedding & Adding {len(chunks)} chunks...")
        
        # Create database in batches with progress bar
        batch_size = 50
        vector_db = None
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="  Embedding & Adding", unit="batch"):
            batch = chunks[i:i+batch_size]
            
            if vector_db is None:
                # Create with first batch
                vector_db = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    persist_directory=chroma_path,
                    collection_name="documents"
                )
            else:
                # Add subsequent batches
                vector_db.add_documents(batch)
    else:
        print(f"  ➕ Adding {len(chunks)} new chunks to existing database...")
        vector_db = Chroma(
            persist_directory=chroma_path,
            embedding_function=embeddings,
            collection_name="documents"
        )
        
        # Add chunks in batches with progress bar
        batch_size = 50
        for i in tqdm(range(0, len(chunks), batch_size), desc="  Embedding & Adding", unit="batch"):
            batch = chunks[i:i+batch_size]
            vector_db.add_documents(batch)
    
    print("  ✅ Database updated successfully")
    
    # Verify final count
    try:
        final_count = vector_db._collection.count()
        print(f"  📊 Total documents in DB: {final_count}")
    except Exception as e:
        print(f"  Could not verify final count: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"\n🎉 Success! Ingestion completed in {elapsed_time:.2f} seconds")
    print(f"📊 New chunks processed: {len(chunks)}")
    if chunks:
        print(f"📊 Sample metadata: {chunks[0].metadata}")
    
    return len(chunks)

if __name__ == "__main__":
    run_ingestion()