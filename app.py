import os
import json
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
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
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = db_client.get_collection(
    name="shl_assessments",
    embedding_function=sentence_transformer_ef
)

# --- 2. Define Strict API Schemas ---
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
async def chat_endpoint(request: ChatRequest):
    messages = request.messages
    if not messages:
        raise HTTPException(status_code=400, detail="Conversation history is empty.")

    # Convert Pydantic messages to dicts for Groq
    conversation = [{"role": m.role, "content": m.content} for m in messages]
    latest_user_message = conversation[-1]["content"]
    
    
    # Convert conversation to a full text transcript for Pass 1 to read
    convo_transcript = "\n".join([f"{m['role']}: {m['content']}" for m in conversation])

    # ==========================================
    # AGENT PASS 1: Search Query Generation
    # ==========================================
    # We now pass the ENTIRE conversation so it remembers previous requirements!
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
    
    # ==========================================
    # RETRIEVAL (ChromaDB)
    # ==========================================
    retrieved_context = "No catalog data retrieved yet."
    if search_intent != "NONE" and search_intent != "":
        # We increased n_results from 5 to 15 to ensure we don't miss secondary skills like SQL or Spring
        results = collection.query(query_texts=[search_intent], n_results=15)
        
        if results['ids'][0]:
            context_parts = []
            for i in range(len(results['ids'][0])):
                meta = results['metadatas'][0][i]
                # We relaxed the distance filter to 1.0 to let more diverse tests through.
                # The Llama 70B model in Pass 2 will filter out the irrelevant ones.
                if results['distances'][0][i] < 1.0: 
                    context_parts.append(
                        f"Name: {meta['name']} | URL: {meta['url']} | Type: {meta['test_type']}"
                    )
            if context_parts:
                retrieved_context = "\n".join(context_parts)
                
                
    # ==========================================
    # AGENT PASS 2: Final Response Generation
    # ==========================================
    # We use JSON mode to guarantee the output matches the SHL schema perfectly.
    system_prompt = f"""
    You are an expert SHL assessment recommender acting as a consultative partner to hiring managers.
    
    CORE BEHAVIORS (Mimic these strictly):
    1. Clarify First: If a request is broad (e.g., "hiring a developer", "contact center staff"), DO NOT recommend immediately. Ask 1-2 targeted questions about seniority, specific daily tasks, or required languages/accents.
    2. Missing Catalog Items: If the user asks for a skill (like 'Rust') that is not in the Retrieved Context, state clearly that it is not in the catalog. Recommend the closest proxy assessments (like general programming or live coding).
    3. Compare & Defend: If asked the difference between two tests, use the Retrieved Context to explain exactly why one fits better (e.g., industry-specific vs general, personality vs knowledge). Explain your reasoning for why you included a test.
    4. Refuse Legal/Compliance Advice: If asked if a test satisfies laws like HIPAA, explicitly state you cannot give legal advice and tell them to consult their legal team.
    5. Maintain State: If the user modifies requirements ("drop OPQ", "add Docker"), update the recommendations list without starting over.
    
    STRICT RULES:
    - ONLY recommend items present in the 'Retrieved Catalog Context' below. Never invent test names or URLs.
    - If clarifying, refusing, or if you don't have enough info yet, the "recommendations" array MUST be empty [].
    - Set "end_of_conversation" to true ONLY when the user explicitly confirms the final list (e.g., "Perfect", "Confirmed", "Lock it in").

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

    # Inject our system prompt at the start of the conversation history
    full_prompt = [{"role": "system", "content": system_prompt}] + conversation

    final_response = groq_client.chat.completions.create(
        messages=full_prompt,
        model="llama-3.3-70b-versatile", # Larger model for strict reasoning and JSON compliance
        temperature=0.1,
        response_format={"type": "json_object"} # Forces valid JSON output
    )

    # Parse the LLM output and return it via FastAPI
    raw_json = final_response.choices[0].message.content
    parsed_response = json.loads(raw_json)
    
    return parsed_response