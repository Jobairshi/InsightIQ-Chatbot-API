from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import requests
import json

# Initialize FastAPI app
app = FastAPI(
    title="Free Chatbot API",
    description="A chatbot API using free alternatives to OpenAI",
    version="1.0.0"
)

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "You are a helpful assistant."

class ChatResponse(BaseModel):
    response: str
    status: str

# Simple mock responses for testing without API costs
def get_mock_response(message: str) -> str:
    """Generate simple mock responses based on keywords"""
    message_lower = message.lower()
    
    if "hello" in message_lower or "hi" in message_lower:
        return "Hello! How can I help you today?"
    elif "joke" in message_lower:
        return "Why don't scientists trust atoms? Because they make up everything!"
    elif "weather" in message_lower:
        return "I'm sorry, I don't have access to real-time weather data, but I hope it's nice where you are!"
    elif "time" in message_lower:
        return "I don't have access to real-time data, but you can check your system clock!"
    elif "python" in message_lower:
        return "Python is a great programming language! It's versatile, readable, and has amazing libraries."
    elif "ai" in message_lower or "artificial intelligence" in message_lower:
        return "AI is fascinating! It's the simulation of human intelligence in machines that are programmed to think and learn."
    else:
        return f"Thank you for your message: '{message}'. This is a mock response since we're avoiding API costs. Your chatbot is working correctly!"

# Alternative: Use Hugging Face's free API (requires internet)
def get_huggingface_response(message: str) -> str:
    """Use Hugging Face's free inference API"""
    try:
        API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
        headers = {"Authorization": "Bearer hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}  # You'd need a free HF token
        
        payload = {
            "inputs": message,
            "parameters": {"max_length": 100, "temperature": 0.7}
        }
        
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            return result[0]['generated_text'] if result else "Sorry, no response generated."
        else:
            return "Hugging Face API unavailable, using mock response."
            
    except Exception as e:
        return f"Error with Hugging Face API: {str(e)}"

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Free Chatbot API is running!", "status": "active", "note": "Using mock responses to avoid API costs"}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "mock-responses", "cost": "free"}

# Main chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Use mock response to avoid API costs
        response_text = get_mock_response(request.message)
        
        return ChatResponse(
            response=response_text,
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Simple chat endpoint
@app.post("/simple-chat")
async def simple_chat(message: str):
    try:
        if not message.strip():
            return {"error": "Message cannot be empty"}
        
        # Use mock response
        response_text = get_mock_response(message)
        
        return {
            "response": response_text,
            "status": "success",
            "note": "Mock response - no API costs"
        }
        
    except Exception as e:
        return {"error": f"Error processing request: {str(e)}"}

# Test endpoint to verify API structure
@app.post("/test")
async def test_endpoint():
    return {
        "message": "API is working correctly!",
        "status": "success",
        "endpoints": [
            "GET /health - Check API status",
            "POST /chat - Main chat with system prompt",
            "POST /simple-chat - Simple chat endpoint",
            "GET /docs - API documentation"
        ]
    }

if __name__ == "__main__":
    print("Starting Free Chatbot API...")
    print("API Documentation will be available at: http://localhost:8001/docs")
    print("Note: Using mock responses to avoid OpenAI API costs")
    uvicorn.run(app, host="0.0.0.0", port=8001)
