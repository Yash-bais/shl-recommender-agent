import chromadb
from chromadb.utils import embedding_functions

def test_retrieval():
    print("Connecting to local ChromaDB...")
    # Initialize the same persistent client
    client = chromadb.PersistentClient(path="./shl_chroma_db")
    
    # using the exact same embedding function we used for ingestion
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    try:
        collection = client.get_collection(
            name="shl_assessments",
            embedding_function=sentence_transformer_ef
        )
    except ValueError:
        print("Error: Collection 'shl_assessments' not found. Make sure the ingestion script ran successfully.")
        return

    # Real life sample queries mimicking what a user might type
    sample_queries = [
        "Hiring a Java developer who works with stakeholders",
        "Mid-level Java dev, around 4 years experience",
        "I need a personality test for a senior leadership role",
        "What do you have for entry-level customer service?"
    ]

    print("\n" + "="*50)
    print("TESTING SEMANTIC RETRIEVAL")
    print("="*50)

    for query in sample_queries:
        print(f"\nUser Query: '{query}'")
        
        # Query the database for the top 3 closest matches
        results = collection.query(
            query_texts=[query],
            n_results=3
        )
        
        # Parse and display the results using our stored metadata
        if not results['ids'][0]:
            print("  No results found.")
            continue
            
        for i in range(len(results['ids'][0])):
            meta = results['metadatas'][0][i]
            # In ChromaDB, a lower distance means a better match.
            distance = results['distances'][0][i] 
            
            print(f"  {i+1}. {meta['name']} (Distance: {distance:.4f})")
            print(f"     Type: {meta['test_type']} | URL: {meta['url']}")

if __name__ == "__main__":
    test_retrieval()