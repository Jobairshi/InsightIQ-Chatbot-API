import requests
import json
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def test_azure_openai():
    """Test Azure OpenAI connection directly"""
    
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
    
    # Construct the full URL
    url = f"{endpoint}openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"
    
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json"
    }
    
    data = {
        "messages": [
            {"role": "user", "content": "Hello! Please respond with a simple greeting to test the Azure OpenAI connection."}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    print("=== Testing Azure OpenAI Connection ===")
    print(f"Endpoint: {endpoint}")
    print(f"Deployment: {deployment_name}")
    print(f"API Version: {api_version}")
    print(f"API Key: {api_key[:20]}...")
    print()
    
    try:
        print("🔄 Sending request to Azure OpenAI...")
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            message = result['choices'][0]['message']['content']
            print(f"🎉 SUCCESS! Azure OpenAI Response:")
            print(f"   {message}")
            print("✅ Your Azure OpenAI is working perfectly!")
            return True
        else:
            print(f"❌ Error Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return False

if __name__ == "__main__":
    test_azure_openai()
