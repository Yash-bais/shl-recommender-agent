import os
import json
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

def build_chroma_db():
    print("Loading data from shl_catalog_scraped.json...")
    with open('shl_catalog_scraped.json', 'r', encoding='utf-8') as f:
        catalog = json.load(f)

    # Initialize ChromaDB with a persistent directory on your local machine
    client = chromadb.PersistentClient(path="./shl_chroma_db")
    
    # Replacing the old sentence_transformer_ef with this:
    hf_ef = embedding_functions.HuggingFaceEmbeddingFunction(
        api_key=os.environ.get("HUGGINGFACE_API_KEY"),
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create or get the collection using the new HF function
    collection = client.get_or_create_collection(
        name="shl_assessments",
        embedding_function=hf_ef
    )

    documents = []
    metadatas = []
    ids = []

    print(f"Preparing {len(catalog)} assessments for embedding...")

    for i, item in enumerate(catalog):
        # 1. Create the Augmented Embedding String
        job_levels_str = ", ".join(item.get('job_levels', []))
        
        # This is the actual text the model will read to understand the assessment
        content = (
            f"Assessment Name: {item['name']}\n"
            f"Test Type: {'Knowledge/Skill' if item['test_type'] == 'K' else 'Personality/Behavioral' if item['test_type'] == 'P' else item['test_type']}\n"
            f"Suitable for Job Levels: {job_levels_str}\n"
            f"Description: {item['description']}"
        )
        
        # 2. Store strict metadata for the final FastAPI JSON schema
        metadata = {
            "name": item['name'],
            "url": item['url'],
            "test_type": item['test_type']
        }
        
        documents.append(content)
        metadatas.append(metadata)
        ids.append(f"doc_{i}")

    print("Embedding and saving to ChromaDB... (This will take a few seconds)")
    
    # Batch add to the database
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print("Successfully built the vector database at ./shl_chroma_db")

if __name__ == "__main__":
    build_chroma_db()