import os
import json
from typing import List, Dict, Any
from fastapi import FastAPI, Request
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# --- 1. Initialize Clients & DB ---
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Connect to your local ChromaDB
db_client = chromadb.PersistentClient(path="./shl_chroma_db")

# Using the Default ONNX Embedding Function as sentence transformer used too much memory and huggingface API keys were a hassle for this demo and it was literally not laoding on my machine maybe because high demand on the embedding endpoint, who knows. 
# (No PyTorch, No API keys, ultra-lightweight!)
onnx_ef = embedding_functions.DefaultEmbeddingFunction()

collection = db_client.get_collection(
    name="shl_assessments",
    embedding_function=onnx_ef
)

# --- 2. Defining Strict API Schemas ---
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool

# Initialize FastAPI
app = FastAPI(title="SHL Recommender Agent")

# --- 3. Endpoints ---

@app.get("/health")
async def health_check():
    """Evaluator readiness check."""
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest): # plugged in Pydantic model!
    # 1. Convert the Pydantic objects into the dictionary list because that's what our code expects
    conversation = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    
    # Format the conversation for the intent extractor
    convo_transcript = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])

    # ==========================================
    # AGENT PASS 1: Search Query Generation
    # ==========================================
    query_prompt = f"""
    You are an AI routing assistant. Read the entire conversation history below.
    Extract ALL active technical skills, job titles, roles, and behavioral requirements into a single, space-separated search string (e.g., "mid-level java spring sql personality stakeholders").
    If the user drops a requirement, do not include it.
    If they are just saying hello and haven't asked for tests, return exactly: NONE.
    
    Conversation History:
    {convo_transcript}
    """
    
    query_response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": query_prompt}],
        model="llama-3.1-8b-instant", 
        temperature=0.0,
        max_tokens=40
    )
    
    search_intent = query_response.choices[0].message.content.strip()
    print(f"Agent Pass 1 Extracted Intent: {search_intent}")

    # ==========================================
    # RETRIEVAL (ChromaDB ONNX Search)
    # ==========================================
    retrieved_context = "No catalog data retrieved yet."
    
    if search_intent != "NONE" and search_intent != "":
        results = collection.query(query_texts=[search_intent], n_results=10)
        
        if results['ids'][0]:
            context_parts = []
            for i in range(len(results['ids'][0])):
                meta = results['metadatas'][0][i]
                doc = results['documents'][0][i]
                
                # Grouping the metadata and the actual description text together!
                context_parts.append(
                    f"Name: {meta['name']} | URL: {meta['url']} | Type: {meta['test_type']}\nDetails: {doc}\n---"
                )
            if context_parts:
                retrieved_context = "\n".join(context_parts)
                
    print(f"Database retrieved {len(retrieved_context)} characters of context.")

    # ==========================================
    # AGENT PASS 2: Final Response Generation
    # ==========================================
    system_prompt = f"""
    You are an expert SHL assessment recommender acting as a consultative partner to hiring managers.
    
    CORE BEHAVIORS (EXECUTE IN THIS EXACT ORDER):
    1. CHECK INVENTORY FIRST: Compare the user's requested skills against the Retrieved Catalog Context. If a specific language, tool, or skill (e.g., Rust, Solidity, Web3) is MISSING from the context, you MUST immediately state that it is not available in our catalog BEFORE asking any clarifying questions. Suggest the closest available alternatives.
    2. CLARIFY BROAD REQUESTS: If the requested skills ARE in the catalog, but the request is vague, ask 1-2 targeted questions about seniority or daily tasks.
    3. COMPARE & DEFEND: Explain exactly why you included a test based on the context.
    4. REFUSE LEGAL ADVICE: If asked about compliance/laws, explicitly state you cannot give legal advice and direct them to their legal team.
    5. MAINTAIN STATE: If the user modifies requirements, update the recommendations.
    
    STRICT RULES:
    - ONLY recommend items present in the 'Retrieved Catalog Context' below. Never invent test names or URLs.
    - If clarifying, refusing, or suggesting alternatives for missing items, the "recommendations" array MUST be empty [].
    - Set "end_of_conversation" to true ONLY when the user explicitly confirms the final list.

    OUTPUT FORMAT:
    You MUST respond in valid JSON matching this schema:
    {{
        "reply": "Your conversational text here",
        "recommendations": [ {{"name": "...", "url": "...", "test_type": "..."}} ],
        "end_of_conversation": boolean
    }}

    Retrieved Catalog Context:
    {retrieved_context}
    """

    # Collecing the system prompt with the user's actual chat history
    full_prompt = [{"role": "system", "content": system_prompt}] + conversation

    final_response = groq_client.chat.completions.create(
        messages=full_prompt,
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        response_format={"type": "json_object"} 
    )

    # Parse the LLM output and return it
    raw_json = final_response.choices[0].message.content
    parsed_response = json.loads(raw_json)
    
    return parsed_response