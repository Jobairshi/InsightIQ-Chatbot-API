import requests
import json

# API base URL
BASE_URL = "http://localhost:8001"

def test_free_chatbot():
    """Test the free chatbot API"""
    
    print("=== Testing Free Chatbot API ===\n")
    
    # Test 1: Health check
    print("1. Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"✅ Health check: {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    print()
    
    # Test 2: Simple chat with various messages
    test_messages = [
        "Hello there!",
        "Tell me a joke",
        "What's the weather like?",
        "Tell me about Python programming",
        "What is artificial intelligence?",
        "Random question about anything"
    ]
    
    print("2. Testing simple chat with various messages...")
    for i, message in enumerate(test_messages, 1):
        try:
            data = {"message": message}
            response = requests.post(f"{BASE_URL}/simple-chat", params=data)
            result = response.json()
            print(f"✅ Test {i}: '{message}'")
            print(f"   Response: {result['response'][:100]}...")
            print()
        except Exception as e:
            print(f"❌ Test {i} failed: {e}")
    
    # Test 3: Advanced chat
    print("3. Testing advanced chat...")
    try:
        data = {
            "message": "Explain machine learning in simple terms",
            "system_prompt": "You are a friendly teacher."
        }
        response = requests.post(f"{BASE_URL}/chat", json=data)
        result = response.json()
        print(f"✅ Advanced chat: {result['response'][:100]}...")
    except Exception as e:
        print(f"❌ Advanced chat failed: {e}")
    print()
    
    # Test 4: Test endpoint
    print("4. Testing API structure...")
    try:
        response = requests.post(f"{BASE_URL}/test")
        result = response.json()
        print(f"✅ Test endpoint: {result['message']}")
        print("Available endpoints:")
        for endpoint in result['endpoints']:
            print(f"   - {endpoint}")
    except Exception as e:
        print(f"❌ Test endpoint failed: {e}")

if __name__ == "__main__":
    try:
        test_free_chatbot()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the API.")
        print("Make sure the server is running: python mail_free.py")
    except Exception as e:
        print(f"❌ Error: {e}")
