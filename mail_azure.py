from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
import uvicorn

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Azure OpenAI Chatbot API",
    description="A chatbot API using Azure OpenAI GPT-4o",
    version="1.0.0"
)

# Initialize Azure OpenAI model
try:
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
        temperature=0.7,
    )
    print("✅ Azure OpenAI initialized successfully!")
except Exception as e:
    print(f"❌ Error initializing Azure OpenAI: {e}")
    llm = None

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "You are a helpful assistant."

class ChatResponse(BaseModel):
    response: str
    status: str

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Azure OpenAI Chatbot API is running!", "status": "active", "model": "gpt-4o"}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "gpt-4o", "provider": "Azure OpenAI"}

# Main chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if not llm:
            raise HTTPException(status_code=500, detail="Azure OpenAI model not initialized")
        
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Create messages for the conversation
        messages = [
            SystemMessage(content=request.system_prompt),
            HumanMessage(content=request.message)
        ]
        
        # Get response from Azure OpenAI
        response = llm.invoke(messages)
        
        return ChatResponse(
            response=response.content,
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Simple chat endpoint without system prompt
@app.post("/simple-chat")
async def simple_chat(message: str):
    try:
        if not llm:
            return {"error": "Azure OpenAI model not initialized"}
        
        if not message.strip():
            return {"error": "Message cannot be empty"}
        
        # Create a simple human message
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=message)
        ]
        
        # Get response from Azure OpenAI
        response = llm.invoke(messages)
        
        return {
            "response": response.content,
            "status": "success",
            "model": "gpt-4o",
            "provider": "Azure OpenAI"
        }
        
    except Exception as e:
        return {"error": f"Error processing request: {str(e)}"}

if __name__ == "__main__":
    print("Starting Azure OpenAI Chatbot API...")
    print("API Documentation will be available at: http://localhost:8000/docs")
    print("Using Azure OpenAI GPT-4o model")
    uvicorn.run(app, host="0.0.0.0", port=8000)
