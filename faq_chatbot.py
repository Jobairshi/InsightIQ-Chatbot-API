from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage, Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import pandas as pd
import os
import uvicorn
from bs4 import BeautifulSoup
import numpy as np
from typing import List, Optional

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Intelligent FAQ Chatbot API",
    description="A chatbot API using Azure OpenAI GPT-4o with custom FAQ knowledge base",
    version="2.0.0"
)

# Global variables for vector store and models
vector_store = None
llm = None
embeddings = None

# Initialize Azure OpenAI models
def initialize_models():
    global llm, embeddings
    try:
        llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
            temperature=0.7,
        )
        
        # Use OpenAI embeddings as fallback (free/cheaper option)
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key= os.getenv("OPENAI_API_KEY"),
        )
        
        print("✅ Azure OpenAI models initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Error initializing Azure OpenAI: {e}")
        return False

# Data processing functions
def clean_html(text):
    """Remove HTML tags from text"""
    if pd.isna(text) or text == "":
        return ""
    return BeautifulSoup(text, "html.parser").get_text()

def load_and_process_data():
    """Load and process FAQ data from CSV files"""
    try:
        # Load CSV files
        faq_df = pd.read_csv("new_features.csv")
        cat_df = pd.read_csv("new_feature_categories.csv")
        
        # Fill NaNs with empty strings
        for df in [faq_df, cat_df]:
            obj_cols = df.select_dtypes(include=["object"]).columns
            df[obj_cols] = df[obj_cols].fillna("")
        
        # Clean HTML from descriptions
        faq_df["desc"] = faq_df["desc"].apply(clean_html)
        
        # Merge with category data
        faq_df = faq_df.merge(
            cat_df[["id", "category_name"]], 
            how="left", 
            left_on="cat_id", 
            right_on="id", 
            suffixes=("", "_cat")
        )
        
        # Combine text for better search
        def combine_text(row):
            return f"""
Title: {row['title']}
Category: {row['category_name']} > {row['sub_cat_name']}
Summary: {row['short_desc']}
Details: {row['desc']}
"""
        
        faq_df["content"] = faq_df.apply(combine_text, axis=1)
        
        print(f"✅ Loaded {len(faq_df)} FAQ entries")
        return faq_df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def create_vector_store(faq_df):
    """Create FAISS vector store from FAQ data"""
    global vector_store, embeddings
    
    if embeddings is None:
        print("❌ Embeddings not initialized")
        return False
    
    try:
        # Create documents
        documents = [
            Document(
                page_content=row["content"], 
                metadata={
                    "id": row["id"], 
                    "title": row["title"],
                    "category": row["category_name"],
                    "sub_category": row["sub_cat_name"]
                }
            )
            for _, row in faq_df.iterrows()
        ]
        
        # Split documents for better embedding
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        split_docs = text_splitter.split_documents(documents)
        
        # Create vector store
        vector_store = FAISS.from_documents(split_docs, embeddings)
        
        # Save for future use
        vector_store.save_local("faiss_faq_index")
        
        print(f"✅ Created vector store with {len(split_docs)} document chunks")
        return True
        
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        return False

def search_faq(query: str, k: int = 3) -> List[dict]:
    """Search FAQ database for relevant entries"""
    global vector_store
    
    if vector_store is None:
        return []
    
    try:
        # Search for relevant documents
        results = vector_store.similarity_search_with_score(query, k=k)
        
        # Format results
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": float(score)
            })
        
        return formatted_results
    except Exception as e:
        print(f"❌ Error searching FAQ: {e}")
        return []

# Initialize everything at startup
def startup():
    """Initialize models and data on startup"""
    print("🚀 Starting Intelligent FAQ Chatbot...")
    
    # Initialize models
    if not initialize_models():
        print("❌ Failed to initialize models")
        return False
    
    # Load and process data
    faq_df = load_and_process_data()
    if faq_df is None:
        print("❌ Failed to load FAQ data")
        return False
    
    # Create vector store
    if not create_vector_store(faq_df):
        print("❌ Failed to create vector store")
        return False
    
    print("✅ Intelligent FAQ Chatbot initialized successfully!")
    return True

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "You are a helpful assistant that answers questions based on FAQ data."
    use_faq: bool = True

class ChatResponse(BaseModel):
    response: str
    status: str
    faq_used: bool = False
    relevant_faqs: Optional[List[dict]] = None

class FAQSearchRequest(BaseModel):
    query: str
    max_results: int = 3

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Intelligent FAQ Chatbot API is running!", 
        "status": "active", 
        "model": "gpt-4o",
        "features": ["FAQ Search", "Intelligent Responses", "Vector Search"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model": "gpt-4o", 
        "provider": "Azure OpenAI",
        "faq_ready": vector_store is not None,
        "embeddings_ready": embeddings is not None
    }

@app.post("/search-faq")
async def search_faq_endpoint(request: FAQSearchRequest):
    """Search FAQ database directly"""
    try:
        results = search_faq(request.query, request.max_results)
        return {
            "query": request.query,
            "results": results,
            "count": len(results),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching FAQ: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def intelligent_chat(request: ChatRequest):
    """Intelligent chat with FAQ integration"""
    try:
        if not llm:
            raise HTTPException(status_code=500, detail="Azure OpenAI model not initialized")
        
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        faq_results = []
        faq_context = ""
        
        # Search FAQ if enabled
        if request.use_faq and vector_store:
            faq_results = search_faq(request.message, k=3)
            
            if faq_results:
                faq_context = "\n\nRelevant FAQ Information:\n"
                for i, result in enumerate(faq_results, 1):
                    faq_context += f"\n{i}. {result['content']}\n"
                faq_context += "\nPlease use this information to provide accurate answers.\n"
        
        # Create enhanced system prompt
        enhanced_system_prompt = f"""{request.system_prompt}

You have access to a knowledge base of FAQ information. When answering questions:
1. First check if the question relates to the provided FAQ information
2. If relevant FAQ information is available, use it as the primary source
3. Provide accurate, helpful answers based on the available information
4. If the question is not covered in the FAQ, provide general helpful information
5. Always be clear about whether your answer comes from the FAQ database or general knowledge

{faq_context}"""
        
        # Create messages
        messages = [
            SystemMessage(content=enhanced_system_prompt),
            HumanMessage(content=request.message)
        ]
        
        # Get response from Azure OpenAI
        response = llm.invoke(messages)
        
        return ChatResponse(
            response=response.content,
            status="success",
            faq_used=len(faq_results) > 0,
            relevant_faqs=faq_results if faq_results else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/simple-chat")
async def simple_chat(message: str, use_faq: bool = True):
    """Simple chat endpoint with FAQ integration"""
    try:
        if not llm:
            return {"error": "Azure OpenAI model not initialized"}
        
        if not message.strip():
            return {"error": "Message cannot be empty"}
        
        faq_results = []
        response_extra = {}
        
        # Search FAQ if enabled
        if use_faq and vector_store:
            faq_results = search_faq(message, k=2)
            response_extra["faq_results"] = faq_results
        
        # Create context-aware prompt
        if faq_results:
            context = "\n\nRelevant FAQ Information:\n"
            for result in faq_results:
                context += f"- {result['content']}\n"
            
            enhanced_message = f"{message}\n{context}\nPlease answer based on the provided FAQ information if relevant."
        else:
            enhanced_message = message
        
        # Create messages
        messages = [
            SystemMessage(content="You are a helpful assistant. Use provided FAQ information when available."),
            HumanMessage(content=enhanced_message)
        ]
        
        # Get response
        response = llm.invoke(messages)
        
        return {
            "response": response.content,
            "status": "success",
            "model": "gpt-4o",
            "provider": "Azure OpenAI",
            "faq_used": len(faq_results) > 0,
            **response_extra
        }
        
    except Exception as e:
        return {"error": f"Error processing request: {str(e)}"}

# Startup event
@app.on_event("startup")
async def startup_event():
    startup()

if __name__ == "__main__":
    print("Starting Intelligent FAQ Chatbot API...")
    print("This may take a moment to load FAQ data and create embeddings...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
