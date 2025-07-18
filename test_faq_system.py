import requests
import json
import time

BASE_URL = "http://localhost:8001"

def test_faq_chatbot():
    """Comprehensive test of the Intelligent FAQ Chatbot API"""
    
    print("=== Testing Intelligent FAQ Chatbot API ===\n")
    
    # Wait for startup
    print("⏳ Waiting for API to initialize (may take 30-60 seconds)...")
    time.sleep(5)
    
    # Test 1: Health check
    print("1. Health Check:")
    try:
        response = requests.get(f"{BASE_URL}/health")
        health = response.json()
        print(f"   ✅ Status: {health['status']}")
        print(f"   ✅ Model: {health['model']}")
        print(f"   ✅ Provider: {health['provider']}")
        print(f"   ✅ FAQ Ready: {health['faq_ready']}")
        print(f"   ✅ Embeddings Ready: {health['embeddings_ready']}")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
    print()
    
    # Test 2: Direct FAQ search
    print("2. Direct FAQ Search:")
    faq_queries = [
        "translation search",
        "instructor control", 
        "header blocks",
        "bug fixes"
    ]
    
    for query in faq_queries:
        try:
            data = {"query": query, "max_results": 2}
            response = requests.post(f"{BASE_URL}/search-faq", json=data)
            result = response.json()
            print(f"   🔍 Query: '{query}'")
            print(f"   📊 Found {result['count']} results")
            if result['results']:
                for i, faq in enumerate(result['results'], 1):
                    title = faq['metadata'].get('title', 'Unknown')
                    score = faq.get('similarity_score', 0)
                    print(f"     {i}. {title} (score: {score:.3f})")
        except Exception as e:
            print(f"   ❌ FAQ search failed for '{query}': {e}")
    print()
    
    # Test 3: Intelligent chat with FAQ
    print("3. Intelligent Chat with FAQ:")
    chat_questions = [
        "How can I search for translation text?",
        "What are the instructor control features?", 
        "Tell me about the new header blocks",
        "What bug fixes were recently made?",
        "How does the translation filtering work?"
    ]
    
    for question in chat_questions:
        try:
            data = {
                "message": question,
                "use_faq": True,
                "system_prompt": "You are an expert support agent who helps users understand features and updates."
            }
            response = requests.post(f"{BASE_URL}/chat", json=data)
            result = response.json()
            print(f"   ❓ Q: {question}")
            print(f"   🤖 A: {result['response'][:120]}...")
            print(f"   📚 FAQ Used: {result['faq_used']}")
            if result['relevant_faqs']:
                print(f"   🔗 Found {len(result['relevant_faqs'])} relevant FAQ entries")
        except Exception as e:
            print(f"   ❌ Chat failed for '{question}': {e}")
        print()
    
    # Test 4: Simple chat with FAQ
    print("4. Simple Chat with FAQ:")
    simple_questions = [
        "Can you help me with translation features?",
        "What's new in the latest update?",
        "How do I use the header designs?"
    ]
    
    for question in simple_questions:
        try:
            response = requests.post(f"{BASE_URL}/simple-chat", 
                                   params={"message": question, "use_faq": True})
            result = response.json()
            print(f"   ❓ Q: {question}")
            print(f"   🤖 A: {result['response'][:100]}...")
            print(f"   📚 FAQ Used: {result['faq_used']}")
        except Exception as e:
            print(f"   ❌ Simple chat failed: {e}")
        print()
    
    # Test 5: Chat without FAQ (general knowledge)
    print("5. General Knowledge (No FAQ):")
    try:
        data = {
            "message": "What is machine learning?",
            "use_faq": False
        }
        response = requests.post(f"{BASE_URL}/chat", json=data)
        result = response.json()
        print(f"   ❓ Q: What is machine learning?")
        print(f"   🤖 A: {result['response'][:120]}...")
        print(f"   📚 FAQ Used: {result['faq_used']}")
    except Exception as e:
        print(f"   ❌ General knowledge test failed: {e}")
    print()
    
    print("🎉 All tests completed!")
    print(f"📱 API Documentation: {BASE_URL}/docs")
    print("🚀 Your Intelligent FAQ Chatbot is ready!")
    print("\n💡 Tips:")
    print("- Use /search-faq for direct FAQ searches")
    print("- Use /chat for intelligent responses with FAQ context")
    print("- Use /simple-chat for quick FAQ-aware responses")

if __name__ == "__main__":
    try:
        test_faq_chatbot()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the API.")
        print("Make sure the server is running: python faq_chatbot.py")
        print("Note: Initial startup may take 30-60 seconds to process FAQ data")
    except Exception as e:
        print(f"❌ Error: {e}")
