import requests
import json
import sys

def run_chat_simulation():
    # The URL where FastAPI server is running
    api_url = "http://127.0.0.1:8000/chat"
    
    # This array acts as the stateless conversation history
    messages = []
    
    print("="*60)
    print("SHL Agent Terminal Simulator")
    print("Type 'exit' or 'quit' to stop.")
    print("="*60)
    
    # We loop until the agent signals the task is done, or the user quits
    while True:
        # 1. Get user input
        user_input = input("\n👤 You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Ending simulation.")
            break
            
        # 2. Append user message to history
        messages.append({"role": "user", "content": user_input})
        
        # 3. Send the full history to the FastAPI endpoint
        payload = {"messages": messages}
        try:
            response = requests.post(api_url, json=payload)
            response.raise_for_status() # Check for HTTP errors
            data = response.json()
        except requests.exceptions.ConnectionError:
            print("\n❌ Error: Could not connect to the server. Is app.py running (uvicorn app:app)?")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print(f"Response content: {response.text}")
            sys.exit(1)
            
        # 4. Extract data from the strict JSON schema
        reply = data.get("reply", "")
        recommendations = data.get("recommendations", [])
        end_of_convo = data.get("end_of_conversation", False)
        
        # 5. Append the agent's reply to the history so it remembers next turn
        messages.append({"role": "assistant", "content": reply})
        
        # 6. Display the agent's response
        print(f"\n🤖 Agent: {reply}")
        
        if recommendations:
            print("\n📋 Recommended Shortlist:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec.get('name')} (Type: {rec.get('test_type')})")
                print(f"     URL: {rec.get('url')}")
        
        # 7. Check if the agent thinks we are finished
        if end_of_convo:
            print("\n✅ Agent signaled: end_of_conversation = True")
            print("Simulation complete.")
            break

if __name__ == "__main__":
    run_chat_simulation()