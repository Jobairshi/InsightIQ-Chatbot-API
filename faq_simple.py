from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
import pandas as pd
import os
import uvicorn
from bs4 import BeautifulSoup
import numpy as np
from typing import List, Optional
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Intelligent FAQ Chatbot API",
    description="A chatbot API using Azure OpenAI GPT-4o with custom FAQ knowledge base",
    version="2.0.0"
)

# Global variables
faq_data = None
vectorizer = None
tfidf_matrix = None
llm = None
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Add this line
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Initialize Azure OpenAI models
def initialize_models():
    global llm
    try:
        llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
            temperature=0.7,
        )
        
        print("✅ Azure OpenAI initialized successfully!")
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

def clean_text_for_search(text):
    """Clean text for better search matching"""
    # Convert to lowercase
    text = str(text).lower()
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def load_and_process_data():
    """Load and process FAQ data from CSV files"""
    global faq_data, vectorizer, tfidf_matrix
    
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
        
        # Create searchable content
        def combine_text(row):
            return f"""
Title: {row['title']}
Category: {row['category_name']} > {row['sub_cat_name']}
Summary: {row['short_desc']}
Details: {row['desc']}
"""
        
        faq_df["content"] = faq_df.apply(combine_text, axis=1)
        faq_df["search_text"] = faq_df["content"].apply(clean_text_for_search)
        
        # Create TF-IDF vectorizer for search
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Fit and transform the FAQ content
        tfidf_matrix = vectorizer.fit_transform(faq_df["search_text"])
        
        faq_data = faq_df
        
        print(f"✅ Loaded {len(faq_df)} FAQ entries")
        print(f"✅ Created TF-IDF search index")
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def search_faq(query: str, k: int = 3) -> List[dict]:
    """Search FAQ database using TF-IDF similarity"""
    global faq_data, vectorizer, tfidf_matrix
    
    if faq_data is None or vectorizer is None or tfidf_matrix is None:
        return []
    
    try:
        # Clean and vectorize the query
        clean_query = clean_text_for_search(query)
        query_vector = vectorizer.transform([clean_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        # Get top k results
        top_indices = similarities.argsort()[-k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                row = faq_data.iloc[idx]
                results.append({
                    "content": row["content"],
                    "metadata": {
                        "id": int(row["id"]),
                        "title": str(row["title"]),
                        "category": str(row["category_name"]),
                        "sub_category": str(row["sub_cat_name"])
                    },
                    "similarity_score": float(similarities[idx])
                })
        
        return results
        
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
    if not load_and_process_data():
        print("❌ Failed to load FAQ data")
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
        "features": ["FAQ Search", "Intelligent Responses", "TF-IDF Search"],
        "faq_count": len(faq_data) if faq_data is not None else 0
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model": "gpt-4o", 
        "provider": "Azure OpenAI",
        "faq_ready": faq_data is not None,
        "search_ready": vectorizer is not None,
        "faq_count": len(faq_data) if faq_data is not None else 0
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
        if request.use_faq and faq_data is not None:
            faq_results = search_faq(request.message, k=3)
            
            if faq_results:
                faq_context = "\n\nRelevant FAQ Information:\n"
                for i, result in enumerate(faq_results, 1):
                    faq_context += f"\n{i}. {result['content']}\n"
                faq_context += "\nPlease use this information to provide accurate answers.\n"
        
        # Create enhanced system prompt
        enhanced_system_prompt = f"""{request.system_prompt}

You have access to a knowledge base of FAQ information about features, updates, and fixes. When answering questions:
1. First check if the question relates to the provided FAQ information
2. If relevant FAQ information is available, use it as the primary source for your answer
3. Provide accurate, helpful answers based on the available information
4. If the question is not covered in the FAQ, provide general helpful information
5. Always be clear and helpful in your responses

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
        if use_faq and faq_data is not None:
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
            SystemMessage(content="You are a helpful assistant that specializes in answering questions about features, updates, and fixes. Use provided FAQ information when available to give accurate answers."),
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
    print("Loading FAQ data and creating search index...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
