import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_chatbot():
    """Test the chatbot API"""
    
    # Test 1: Health check
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health check: {response.json()}")
    print()
    
    # Test 2: Simple chat
    print("Testing simple chat...")
    data = {"message": "Hello! Tell me a joke."}
    response = requests.post(f"{BASE_URL}/simple-chat", params=data)
    print(f"Simple chat response: {response.json()}")
    print()
    
    # Test 3: Advanced chat with custom system prompt
    print("Testing advanced chat...")
    data = {
        "message": "Explain quantum computing in simple terms",
        "system_prompt": "You are a science teacher explaining complex topics to high school students."
    }
    response = requests.post(f"{BASE_URL}/chat", json=data)
    print(f"Advanced chat response: {response.json()}")
    print()

if __name__ == "__main__":
    try:
        test_chatbot()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"Error: {e}")
