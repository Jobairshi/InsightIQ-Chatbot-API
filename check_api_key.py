import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

def test_api_key():
    """Test if the OpenAI API key is valid and has credits"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    print("=== OpenAI API Key Test ===")
    print(f"API Key: {api_key[:20]}...{api_key[-10:]}")  # Show partial key for verification
    print()
    
    try:
        # Try to initialize the model
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=api_key
        )
        print("✅ Model initialized successfully")
        
        # Try a minimal test call
        from langchain.schema import HumanMessage
        messages = [HumanMessage(content="Say 'test'")]
        
        print("🔄 Testing API call...")
        response = llm.invoke(messages)
        print(f"✅ SUCCESS! API Response: {response.content}")
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        
        if "insufficient_quota" in str(e):
            print("\n💡 SOLUTION:")
            print("1. Go to https://platform.openai.com/billing")
            print("2. Add a payment method (credit card)")
            print("3. Add at least $5 credit to your account")
            print("4. Or use the free mock version (mail_free.py)")
        elif "invalid" in str(e).lower():
            print("\n💡 Your API key might be invalid or expired")
        else:
            print(f"\n💡 Unknown error: {str(e)}")

if __name__ == "__main__":
    test_api_key()
