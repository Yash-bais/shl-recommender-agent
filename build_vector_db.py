import json
import chromadb
from chromadb.utils import embedding_functions

def build_chroma_db():
    # 1. Initialize Chroma normally
    client = chromadb.PersistentClient(path="./shl_chroma_db")
    
    # 2. Using the Default ONNX Embedding Function 
    onnx_ef = embedding_functions.DefaultEmbeddingFunction()

    collection = client.get_or_create_collection(
        name="shl_assessments",
        embedding_function=onnx_ef
    )

    documents = []
    metadatas = []
    ids = []

    print("Loading data from shl_catalog_scraped.json...")
    with open("shl_catalog_scraped.json", "r", encoding="utf-8") as f:
        catalog = json.load(f)
        
    if not catalog:
        raise ValueError("❌ The JSON file is empty!")

    print(f"Preparing {len(catalog)} assessments for embedding...")

    for i, item in enumerate(catalog):
        job_levels_str = ", ".join(item.get('job_levels', []))
        
        content = (
            f"Assessment Name: {item['name']}\n"
            f"Test Type: {'Knowledge/Skill' if item['test_type'] == 'K' else 'Personality/Behavioral' if item['test_type'] == 'P' else item['test_type']}\n"
            f"Suitable for Job Levels: {job_levels_str}\n"
            f"Description: {item['description']}"
        )
        
        metadata = {
            "name": item['name'],
            "url": item['url'],
            "test_type": item['test_type']
        }
        
        documents.append(content)
        metadatas.append(metadata)
        ids.append(f"doc_{i}")

    print("Embedding locally using ONNX runtime... (Lightning fast!)")
    
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print("Successfully built the vector database at ./shl_chroma_db")


if __name__ == "__main__":
    build_chroma_db()