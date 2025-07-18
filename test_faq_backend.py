import requests
import json
import time

# FAQ Backend runs on port 8001
FAQ_BASE_URL = "http://localhost:8001"

def test_faq_backend():
    """Comprehensive test of the FAQ backend API"""
    
    print("=== Testing FAQ Backend API ===\n")
    
    # Test 1: Health check
    print("1. Health Check:")
    try:
        response = requests.get(f"{FAQ_BASE_URL}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"   ✅ Status: {health['status']}")
            print(f"   ✅ Model: {health.get('model', 'N/A')}")
            print(f"   ✅ Provider: {health.get('provider', 'N/A')}")
            print(f"   ✅ FAQ Records: {health.get('faq_records', 'N/A')}")
        else:
            print(f"   ❌ Health check failed with status: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Connection failed - Backend not running on {FAQ_BASE_URL}")
        return False
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        return False
    print()
    
    # Test 2: FAQ Search API
    print("2. FAQ Search Test:")
    test_queries = [
        "translation search",
        "header blocks",
        "instructor control",
        "bug fixes",
        "new features"
    ]
    
    for query in test_queries:
        try:
            data = {
                "query": query,
                "max_results": 3
            }
            response = requests.post(f"{FAQ_BASE_URL}/search-faq", json=data)
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Query: '{query}'")
                print(f"       Results found: {result.get('count', 0)}")
                if result.get('results'):
                    top_result = result['results'][0]
                    print(f"       Top match: {top_result.get('similarity_score', 0):.2f} similarity")
                    print(f"       Category: {top_result.get('metadata', {}).get('category', 'N/A')}")
            else:
                print(f"   ❌ Search failed for '{query}' with status: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Search failed for '{query}': {e}")
    print()
    
    # Test 3: AI Chat API
    print("3. AI Chat Test:")
    chat_questions = [
        "What are the latest feature updates?",
        "Tell me about translation search functionality",
        "How do I use header blocks?",
        "What bug fixes were released recently?"
    ]
    
    for question in chat_questions:
        try:
            data = {
                "message": question,
                "system_prompt": "You are a helpful FAQ assistant. Use the provided FAQ context to answer questions accurately."
            }
            response = requests.post(f"{FAQ_BASE_URL}/chat", json=data)
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Question: '{question}'")
                print(f"       Response: {result.get('response', '')[:100]}...")
                print(f"       FAQ Used: {result.get('faq_used', False)}")
                if result.get('usage'):
                    usage = result['usage']
                    print(f"       Tokens: {usage.get('total_tokens', 0)} total")
            else:
                print(f"   ❌ Chat failed for '{question}' with status: {response.status_code}")
                print(f"       Error: {response.text}")
        except Exception as e:
            print(f"   ❌ Chat failed for '{question}': {e}")
        
        # Small delay between requests to avoid rate limiting
        time.sleep(1)
    print()
    
    # Test 4: Simple Chat API
    print("4. Simple Chat Test:")
    simple_questions = [
        "Hello, how can you help me?",
        "What is this system about?",
        "Can you explain your capabilities?"
    ]
    
    for question in simple_questions:
        try:
            response = requests.post(f"{FAQ_BASE_URL}/simple-chat", params={"message": question})
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Question: '{question}'")
                print(f"       Response: {result.get('response', '')[:80]}...")
                print(f"       Status: {result.get('status', 'N/A')}")
            else:
                print(f"   ❌ Simple chat failed for '{question}' with status: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Simple chat failed for '{question}': {e}")
    print()
    
    # Test 5: API Documentation
    print("5. API Documentation Test:")
    try:
        response = requests.get(f"{FAQ_BASE_URL}/docs")
        if response.status_code == 200:
            print(f"   ✅ API docs available at: {FAQ_BASE_URL}/docs")
        else:
            print(f"   ❌ API docs not accessible")
    except Exception as e:
        print(f"   ❌ API docs test failed: {e}")
    print()
    
    # Test 6: Error Handling
    print("6. Error Handling Test:")
    try:
        # Test with empty query
        response = requests.post(f"{FAQ_BASE_URL}/search-faq", json={"query": "", "max_results": 5})
        print(f"   Empty query test: Status {response.status_code}")
        
        # Test with invalid JSON
        response = requests.post(f"{FAQ_BASE_URL}/chat", json={"invalid": "data"})
        print(f"   Invalid data test: Status {response.status_code}")
        
    except Exception as e:
        print(f"   ❌ Error handling test failed: {e}")
    print()
    
    print("🎉 FAQ Backend API Testing Complete!")
    print(f"📱 API Documentation: {FAQ_BASE_URL}/docs")
    print(f"🔍 FAQ Search Endpoint: {FAQ_BASE_URL}/search-faq")
    print(f"💬 Chat Endpoint: {FAQ_BASE_URL}/chat")
    print(f"🚀 Simple Chat Endpoint: {FAQ_BASE_URL}/simple-chat")
    return True

def check_backend_status():
    """Quick check if backend is running"""
    try:
        response = requests.get(f"{FAQ_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

if __name__ == "__main__":
    print("Checking if FAQ backend is running...")
    if not check_backend_status():
        print(f"❌ FAQ Backend is not running on {FAQ_BASE_URL}")
        print("Please start the backend first using:")
        print("   python faq_simple.py")
        print()
    else:
        print("✅ Backend is running, starting tests...\n")
        test_faq_backend()
