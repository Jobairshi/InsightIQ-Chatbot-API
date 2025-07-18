from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
import uvicorn

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Hybrid Chatbot API",
    description="A chatbot API that uses OpenAI when available, falls back to mock responses",
    version="1.0.0"
)

# Initialize OpenAI model
try:
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    # Test the connection
    test_messages = [SystemMessage(content="Test"), HumanMessage(content="Hi")]
    test_response = llm.invoke(test_messages)
    openai_available = True
    print("✅ OpenAI API connected successfully!")
except Exception as e:
    print(f"⚠️ OpenAI API not available: {e}")
    llm = None
    openai_available = False

# Fallback responses for when OpenAI is not available
def get_fallback_response(message: str) -> str:
    """Generate fallback responses when OpenAI is not available"""
    message_lower = message.lower()
    
    responses = {
        "hello": "Hello! I'm a fallback chatbot. OpenAI credits are needed for full AI responses.",
        "hi": "Hi there! I'm running in fallback mode due to OpenAI quota limits.",
        "joke": "Why did the programmer quit? Because they didn't get arrays! (Fallback joke)",
        "weather": "I can't check weather in fallback mode, but I hope it's nice!",
        "python": "Python is awesome! It's great for APIs like this one.",
        "ai": "AI is fascinating! This API normally uses OpenAI, but is in fallback mode.",
        "help": "I'm in fallback mode. For full AI responses, OpenAI credits are needed.",
        "how are you": "I'm doing well in fallback mode! How can I help?",
        "what can you do": "In fallback mode, I can give basic responses. With OpenAI credits, I'd be much smarter!"
    }
    
    # Check for keywords
    for keyword, response in responses.items():
        if keyword in message_lower:
            return response
    
    # Default response
    return f"Thanks for your message! I'm currently in fallback mode due to OpenAI quota limits. Your message was: '{message}'. For full AI responses, please add OpenAI credits to your account."

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "You are a helpful assistant."

class ChatResponse(BaseModel):
    response: str
    status: str
    mode: str  # "openai" or "fallback"

# Root endpoint
@app.get("/")
async def root():
    mode = "OpenAI" if openai_available else "Fallback"
    return {
        "message": f"Hybrid Chatbot API is running in {mode} mode!", 
        "status": "active",
        "openai_available": openai_available
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model": "gpt-3.5-turbo" if openai_available else "fallback-responses",
        "openai_available": openai_available,
        "mode": "OpenAI" if openai_available else "Fallback"
    }

# Main chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Try OpenAI first, fall back if it fails
        if openai_available and llm:
            try:
                messages = [
                    SystemMessage(content=request.system_prompt),
                    HumanMessage(content=request.message)
                ]
                response = llm.invoke(messages)
                return ChatResponse(
                    response=response.content,
                    status="success",
                    mode="openai"
                )
            except Exception as e:
                # If OpenAI fails, use fallback
                fallback_response = get_fallback_response(request.message)
                return ChatResponse(
                    response=f"{fallback_response}\n\n[OpenAI Error: {str(e)}]",
                    status="fallback",
                    mode="fallback"
                )
        else:
            # Use fallback response
            fallback_response = get_fallback_response(request.message)
            return ChatResponse(
                response=fallback_response,
                status="fallback",
                mode="fallback"
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Simple chat endpoint
@app.post("/simple-chat")
async def simple_chat(message: str):
    try:
        if not message.strip():
            return {"error": "Message cannot be empty"}
        
        # Try OpenAI first, fall back if it fails
        if openai_available and llm:
            try:
                messages = [
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content=message)
                ]
                response = llm.invoke(messages)
                return {
                    "response": response.content,
                    "status": "success",
                    "mode": "openai"
                }
            except Exception as e:
                # If OpenAI fails, use fallback
                fallback_response = get_fallback_response(message)
                return {
                    "response": fallback_response,
                    "status": "fallback",
                    "mode": "fallback",
                    "openai_error": str(e)
                }
        else:
            # Use fallback response
            fallback_response = get_fallback_response(message)
            return {
                "response": fallback_response,
                "status": "fallback",
                "mode": "fallback"
            }
        
    except Exception as e:
        return {"error": f"Error processing request: {str(e)}"}

if __name__ == "__main__":
    mode = "OpenAI" if openai_available else "Fallback"
    print(f"Starting Hybrid Chatbot API in {mode} mode...")
    print("API Documentation will be available at: http://localhost:8000/docs")
    if not openai_available:
        print("💡 Tip: Add OpenAI credits or create new account for full AI responses!")
    uvicorn.run(app, host="0.0.0.0", port=8002)
