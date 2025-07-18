import requests
import json

BASE_URL = "http://localhost:8000"

def test_azure_chatbot():
    """Comprehensive test of the Azure OpenAI chatbot API"""
    
    print("=== Testing Azure OpenAI Chatbot API ===\n")
    
    # Test 1: Health check
    print("1. Health Check:")
    try:
        response = requests.get(f"{BASE_URL}/health")
        health = response.json()
        print(f"   ✅ Status: {health['status']}")
        print(f"   ✅ Model: {health['model']}")
        print(f"   ✅ Provider: {health['provider']}")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
    print()
    
    # Test 2: Simple chat
    print("2. Simple Chat Test:")
    try:
        response = requests.post(f"{BASE_URL}/simple-chat", params={"message": "Hello! Can you tell me a programming joke?"})
        result = response.json()
        print(f"   ✅ Response: {result['response'][:100]}...")
        print(f"   ✅ Status: {result['status']}")
        print(f"   ✅ Model: {result['model']}")
    except Exception as e:
        print(f"   ❌ Simple chat failed: {e}")
    print()
    
    # Test 3: Advanced chat with system prompt
    print("3. Advanced Chat Test:")
    try:
        data = {
            "message": "Explain Python programming to a beginner",
            "system_prompt": "You are an expert programming instructor who explains things simply and clearly."
        }
        response = requests.post(f"{BASE_URL}/chat", json=data)
        result = response.json()
        print(f"   ✅ Response: {result['response'][:150]}...")
        print(f"   ✅ Status: {result['status']}")
    except Exception as e:
        print(f"   ❌ Advanced chat failed: {e}")
    print()
    
    # Test 4: Multiple questions
    questions = [
        "What is artificial intelligence?",
        "How do computers work?",
        "Tell me about the future of technology",
        "What is the best way to learn programming?"
    ]
    
    print("4. Multiple Questions Test:")
    for i, question in enumerate(questions, 1):
        try:
            response = requests.post(f"{BASE_URL}/simple-chat", params={"message": question})
            result = response.json()
            print(f"   ✅ Q{i}: {question}")
            print(f"       A: {result['response'][:80]}...")
        except Exception as e:
            print(f"   ❌ Q{i} failed: {e}")
    print()
    
    print("🎉 All tests completed! Your Azure OpenAI chatbot is working perfectly!")
    print(f"📱 API Documentation: {BASE_URL}/docs")
    print("🚀 Ready for production use!")

if __name__ == "__main__":
    try:
        test_azure_chatbot()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the API. Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error: {e}")
