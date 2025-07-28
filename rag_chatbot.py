from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pandas as pd
import os
import uvicorn
from bs4 import BeautifulSoup
import numpy as np
from typing import List, Optional, Dict, Any
import re
import pinecone
from pinecone import Pinecone
import hashlib
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Advanced RAG-Powered FAQ Chatbot API",
    description="A sophisticated chatbot API using Azure OpenAI GPT-4o with Pinecone vector database and RAG implementation",
    version="3.0.0"
)

# Global variables
faq_data = None
llm = None
embeddings_model = None
pinecone_client = None
vector_store = None
qa_chain = None

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
PINECONE_API_KEY = "pcsk_4BnTBd_MWr6WTcLR1FH7MCATTLbKWMBah8becVf6KUecVUmzc5usoNsTjY6gQd2EGqNvVC"
PINECONE_INDEX_NAME = "faq-knowledge-base"
PINECONE_DIMENSION = 1536  # Azure OpenAI text-embedding-ada-002 dimension
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Initialize Pinecone
def initialize_pinecone():
    """Initialize Pinecone client and create/connect to index"""
    global pinecone_client
    
    try:
        # Initialize Pinecone client (v2.x style)
        pinecone.init(
            api_key=PINECONE_API_KEY,
            environment="us-east-1"
        )
        
        # Check if index exists, create if not
        existing_indexes = pinecone.list_indexes()
        
        if PINECONE_INDEX_NAME not in existing_indexes:
            logger.info(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
            pinecone.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=PINECONE_DIMENSION,
                metric="cosine"
            )
            logger.info("✅ Pinecone index created successfully!")
        else:
            logger.info(f"✅ Connected to existing Pinecone index: {PINECONE_INDEX_NAME}")
        
        # Get the index
        pinecone_client = pinecone.Index(PINECONE_INDEX_NAME)
        return True
        
    except Exception as e:
        logger.error(f"❌ Error initializing Pinecone: {e}")
        return False

# Initialize Azure OpenAI models
def initialize_models():
    """Initialize Azure OpenAI LLM and Embeddings models"""
    global llm, embeddings_model
    
    try:
        # Initialize LLM
        llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
            temperature=0.7,
            max_tokens=2000,
        )
        
        # Initialize Embeddings model
        embeddings_model = AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_deployment="text-embedding-ada-002",  # Common embedding deployment name
            chunk_size=1000,
        )
        
        logger.info("✅ Azure OpenAI models initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error initializing Azure OpenAI models: {e}")
        return False

# Data processing functions
def clean_html(text):
    """Remove HTML tags from text"""
    if pd.isna(text) or text == "":
        return ""
    return BeautifulSoup(text, "html.parser").get_text()

def create_document_chunks(faq_df):
    """Create document chunks for vector storage"""
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    for _, row in faq_df.iterrows():
        # Create comprehensive document content
        content = f"""
Title: {row['title']}
Category: {row['category_name']} > {row['sub_cat_name']}
Summary: {row['short_desc']}
Detailed Description: {row['desc']}
"""
        
        # Create metadata
        metadata = {
            "id": str(row['id']),
            "title": str(row['title']),
            "category": str(row['category_name']),
            "sub_category": str(row['sub_cat_name']),
            "short_desc": str(row['short_desc']),
            "source": "FAQ_Database",
            "chunk_type": "complete_entry"
        }
        
        # Split into chunks if content is too long
        chunks = text_splitter.split_text(content)
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = f"{row['id']}_{i}"
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(chunks)
            
            documents.append({
                "content": chunk,
                "metadata": chunk_metadata
            })
    
    return documents

def load_and_process_data():
    """Load FAQ data and create vector embeddings in Pinecone"""
    global faq_data, vector_store, qa_chain
    
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
        
        faq_data = faq_df
        logger.info(f"✅ Loaded {len(faq_df)} FAQ entries")
        
        # Create document chunks
        documents = create_document_chunks(faq_df)
        logger.info(f"✅ Created {len(documents)} document chunks")
        
        # Initialize vector store with Pinecone
        vector_store = LangchainPinecone.from_texts(
            texts=[doc["content"] for doc in documents],
            metadatas=[doc["metadata"] for doc in documents],
            embedding=embeddings_model,
            index_name=PINECONE_INDEX_NAME
        )
        
        logger.info("✅ Vector store created and populated with embeddings")
        
        # Create QA Chain with custom prompt
        qa_prompt = PromptTemplate(
            template="""You are an intelligent FAQ assistant with access to a comprehensive knowledge base. 
Use the following context to answer the question accurately and comprehensively.

Context from FAQ Database:
{context}

Question: {question}

Instructions:
1. Base your answer primarily on the provided context
2. If the context doesn't contain relevant information, clearly state that
3. Provide specific details when available
4. If multiple related topics are found, organize your response clearly
5. Be helpful and comprehensive in your responses

Answer:""",
            input_variables=["context", "question"]
        )
        
        # Create RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 5,
                    "score_threshold": 0.7
                }
            ),
            chain_type_kwargs={"prompt": qa_prompt},
            return_source_documents=True
        )
        
        logger.info("✅ RAG QA Chain created successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading and processing data: {e}")
        return False

def search_vector_db(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Search vector database using semantic similarity"""
    global vector_store
    
    if vector_store is None:
        return []
    
    try:
        # Perform similarity search with metadata
        results = vector_store.similarity_search_with_score(query, k=k)
        
        search_results = []
        for doc, score in results:
            result = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": float(1 - score),  # Convert distance to similarity
                "relevance": "high" if (1 - score) > 0.8 else "medium" if (1 - score) > 0.6 else "low"
            }
            search_results.append(result)
        
        return search_results
        
    except Exception as e:
        logger.error(f"❌ Error searching vector database: {e}")
        return []

def get_rag_response(query: str) -> Dict[str, Any]:
    """Get response using RAG with source documents"""
    global qa_chain
    
    if qa_chain is None:
        return {"error": "RAG system not initialized"}
    
    try:
        # Get response from QA chain
        result = qa_chain({"query": query})
        
        # Process source documents
        source_docs = []
        if "source_documents" in result:
            for doc in result["source_documents"]:
                source_docs.append({
                    "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "metadata": doc.metadata,
                    "source": doc.metadata.get("source", "Unknown")
                })
        
        return {
            "answer": result["result"],
            "source_documents": source_docs,
            "total_sources": len(source_docs),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting RAG response: {e}")
        return {"error": f"Error processing query: {str(e)}"}

# Fine-tuning data preparation
def prepare_fine_tuning_data():
    """Prepare data for GPT fine-tuning in OpenAI format"""
    global faq_data
    
    if faq_data is None:
        return None
    
    try:
        fine_tuning_data = []
        
        for _, row in faq_data.iterrows():
            # Create training examples in OpenAI format
            training_example = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful FAQ assistant that provides accurate information about features, updates, and fixes based on your knowledge base."
                    },
                    {
                        "role": "user",
                        "content": f"What can you tell me about {row['title']}?"
                    },
                    {
                        "role": "assistant",
                        "content": f"**{row['title']}**\n\nCategory: {row['category_name']} > {row['sub_cat_name']}\n\nSummary: {row['short_desc']}\n\nDetails: {row['desc']}"
                    }
                ]
            }
            fine_tuning_data.append(training_example)
            
            # Create variation with different question format
            if len(row['short_desc']) > 10:
                variation_example = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful FAQ assistant that provides accurate information about features, updates, and fixes based on your knowledge base."
                        },
                        {
                            "role": "user",
                            "content": f"Can you explain {row['short_desc'].lower()}?"
                        },
                        {
                            "role": "assistant",
                            "content": f"Regarding {row['short_desc']}: {row['desc']}\n\nThis is part of the {row['category_name']} category and relates to {row['title']}."
                        }
                    ]
                }
                fine_tuning_data.append(variation_example)
        
        return fine_tuning_data
        
    except Exception as e:
        logger.error(f"❌ Error preparing fine-tuning data: {e}")
        return None

# Initialize everything at startup
def startup():
    """Initialize all components on startup"""
    logger.info("🚀 Starting Advanced RAG-Powered FAQ Chatbot...")
    
    # Initialize Pinecone
    if not initialize_pinecone():
        logger.error("❌ Failed to initialize Pinecone")
        return False
    
    # Initialize models
    if not initialize_models():
        logger.error("❌ Failed to initialize models")
        return False
    
    # Load and process data
    if not load_and_process_data():
        logger.error("❌ Failed to load FAQ data")
        return False
    
    logger.info("✅ Advanced RAG-Powered FAQ Chatbot initialized successfully!")
    return True

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "You are a helpful assistant that answers questions based on FAQ data."
    use_rag: bool = True

class ChatResponse(BaseModel):
    response: str
    status: str
    rag_used: bool = False
    source_documents: Optional[List[dict]] = None
    retrieval_score: Optional[float] = None

class RAGSearchRequest(BaseModel):
    query: str
    max_results: int = 5
    score_threshold: float = 0.6

class RAGSearchResponse(BaseModel):
    query: str
    results: List[dict]
    total_results: int
    status: str

class CourseLessonRequest(BaseModel):
    topic: str
    difficulty_level: str = "intermediate"
    lesson_duration: str = "60 minutes"
    include_examples: bool = True
    include_exercises: bool = True

class CourseLessonResponse(BaseModel):
    topic: str
    lesson_content: str
    status: str
    difficulty_level: str
    estimated_duration: str

class MCQGeneratorRequest(BaseModel):
    topic: str
    number_of_questions: int
    difficulty_level: str = "intermediate"

class MCQQuestion(BaseModel):
    question: str
    options: List[str]
    correct_answer: str

class MCQGeneratorResponse(BaseModel):
    topic: str
    questions: List[MCQQuestion]
    total_questions: int
    status: str

class FineTuningDataResponse(BaseModel):
    total_examples: int
    sample_data: List[dict]
    download_ready: bool
    status: str

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Advanced RAG-Powered FAQ Chatbot API is running!", 
        "status": "active", 
        "model": "gpt-4o",
        "features": [
            "Pinecone Vector Database", 
            "RAG Implementation", 
            "Semantic Search", 
            "Course Generator", 
            "MCQ Generator",
            "Fine-tuning Data Preparation"
        ],
        "faq_count": len(faq_data) if faq_data is not None else 0,
        "vector_store_ready": vector_store is not None,
        "rag_chain_ready": qa_chain is not None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model": "gpt-4o", 
        "provider": "Azure OpenAI",
        "vector_db": "Pinecone",
        "faq_ready": faq_data is not None,
        "vector_store_ready": vector_store is not None,
        "rag_ready": qa_chain is not None,
        "faq_count": len(faq_data) if faq_data is not None else 0
    }

@app.post("/rag-search", response_model=RAGSearchResponse)
async def rag_search_endpoint(request: RAGSearchRequest):
    """Search using RAG vector database"""
    try:
        results = search_vector_db(request.query, request.max_results)
        
        # Filter by score threshold
        filtered_results = [
            result for result in results
            if result["similarity_score"] >= request.score_threshold
        ]
        
        return RAGSearchResponse(
            query=request.query,
            results=filtered_results,
            total_results=len(filtered_results),
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing RAG search: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def intelligent_chat(request: ChatRequest):
    """Advanced chat with RAG implementation"""
    try:
        if not llm:
            raise HTTPException(status_code=500, detail="Azure OpenAI model not initialized")
        
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        if request.use_rag and qa_chain:
            # Use RAG for response
            rag_result = get_rag_response(request.message)
            
            if "error" in rag_result:
                raise HTTPException(status_code=500, detail=rag_result["error"])
            
            return ChatResponse(
                response=rag_result["answer"],
                status="success",
                rag_used=True,
                source_documents=rag_result["source_documents"],
                retrieval_score=0.8  # You can implement actual scoring
            )
        else:
            # Fallback to regular chat
            messages = [
                SystemMessage(content=request.system_prompt),
                HumanMessage(content=request.message)
            ]
            response = llm.invoke(messages)
            
            return ChatResponse(
                response=response.content,
                status="success",
                rag_used=False
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/generate-course-lesson", response_model=CourseLessonResponse)
async def generate_course_lesson(request: CourseLessonRequest):
    """Generate course lesson with RAG context if available"""
    try:
        if not llm:
            raise HTTPException(status_code=500, detail="Azure OpenAI model not initialized")
        
        # Get relevant context from RAG if available
        context = ""
        if vector_store:
            search_results = search_vector_db(request.topic, k=3)
            if search_results:
                context = "\n\nRelevant knowledge base information:\n"
                for result in search_results[:2]:  # Use top 2 results
                    context += f"- {result['content'][:300]}...\n"
        
        system_prompt = f"""You are an expert course instructor and curriculum designer. Create a comprehensive, well-structured lesson on the given topic.
Guidelines for lesson creation:
1. Structure the lesson with clear sections: Introduction, Learning Objectives, Main Content, Key Concepts, and Summary
2. Make content appropriate for {request.difficulty_level} level learners
3. Design for approximately {request.lesson_duration} of learning time
4. Use clear, engaging language with proper explanations
5. Include practical insights and real-world applications
6. Organize content in a logical, progressive manner

{"Include relevant examples and case studies to illustrate concepts." if request.include_examples else "Focus on theoretical concepts without specific examples."}
{"Include practice exercises and activities for hands-on learning." if request.include_exercises else "Keep content focused on conceptual learning without exercises."}

{context}

Create a complete, ready-to-teach lesson that covers all essential aspects of the topic."""
        
        lesson_prompt = f"Create a comprehensive course lesson on the topic: '{request.topic}'"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=lesson_prompt)
        ]
        
        response = llm.invoke(messages)
        
        return CourseLessonResponse(
            topic=request.topic,
            lesson_content=response.content,
            status="success",
            difficulty_level=request.difficulty_level,
            estimated_duration=request.lesson_duration
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating course lesson: {str(e)}")

@app.post("/generate-mcq", response_model=MCQGeneratorResponse)
async def generate_mcq(request: MCQGeneratorRequest):
    """Generate MCQ with enhanced context from RAG"""
    try:
        if not llm:
            raise HTTPException(status_code=500, detail="Azure OpenAI model not initialized")
        
        if request.number_of_questions <= 0 or request.number_of_questions > 50:
            raise HTTPException(status_code=400, detail="Number of questions must be between 1 and 50")
        
        # Get relevant context from knowledge base
        context = ""
        if vector_store:
            search_results = search_vector_db(request.topic, k=5)
            if search_results:
                context = "\n\nUse the following knowledge base information to create more accurate questions:\n"
                for result in search_results[:3]:
                    context += f"- {result['content'][:400]}...\n"
        
        system_prompt = f"""You are an expert assessment designer. Generate exactly {request.number_of_questions} multiple choice questions on the given topic.
CRITICAL REQUIREMENTS:
1. Generate EXACTLY {request.number_of_questions} questions
2. Each question must have exactly 4 options (A, B, C, D)
3. Questions should be at {request.difficulty_level} difficulty level
4. Provide clear, unambiguous questions
5. Include one correct answer and three plausible distractors
6. Return ONLY the questions and options in the specified format below
7. NO additional text, explanations, or commentary

{context}

FORMAT REQUIREMENT - Return EXACTLY in this JSON format:
{{
  "questions": [
    {{
      "question": "Your question here?",
      "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
      "correct_answer": "A) Option 1"
    }}
  ]
}}

Generate questions that test understanding, application, and analysis of the topic."""
        
        mcq_prompt = f"Generate {request.number_of_questions} multiple choice questions on the topic: '{request.topic}'"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=mcq_prompt)
        ]
        
        response = llm.invoke(messages)
        
        # Parse response (same logic as before)
        try:
            import json
            response_text = response.content.strip()
            
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            elif response_text.startswith("```"):
                response_text = response_text.replace("```", "").strip()
            
            parsed_response = json.loads(response_text)
            
            if "questions" in parsed_response:
                questions_data = parsed_response["questions"]
            else:
                questions_data = parsed_response
            
            mcq_questions = []
            for q_data in questions_data:
                mcq_question = MCQQuestion(
                    question=q_data["question"],
                    options=q_data["options"],
                    correct_answer=q_data["correct_answer"]
                )
                mcq_questions.append(mcq_question)
            
            return MCQGeneratorResponse(
                topic=request.topic,
                questions=mcq_questions,
                total_questions=len(mcq_questions),
                status="success"
            )
            
        except json.JSONDecodeError:
            # Fallback parsing logic remains the same
            lines = response.content.split('\n')
            questions = []
            current_question = None
            current_options = []
            current_correct = ""
            
            for line in lines:
                line = line.strip()
                if line and '?' in line and not line.startswith(('A)', 'B)', 'C)', 'D)')):
                    if current_question and current_options:
                        questions.append(MCQQuestion(
                            question=current_question,
                            options=current_options,
                            correct_answer=current_correct
                        ))
                    
                    current_question = line
                    current_options = []
                    current_correct = ""
                elif line.startswith(('A)', 'B)', 'C)', 'D)')):
                    current_options.append(line)
                    if not current_correct:
                        current_correct = line
            
            if current_question and current_options:
                questions.append(MCQQuestion(
                    question=current_question,
                    options=current_options,
                    correct_answer=current_correct
                ))
            
            return MCQGeneratorResponse(
                topic=request.topic,
                questions=questions,
                total_questions=len(questions),
                status="success"
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating MCQ: {str(e)}")

@app.get("/fine-tuning-data", response_model=FineTuningDataResponse)
async def get_fine_tuning_data():
    """Get fine-tuning data preparation for GPT model"""
    try:
        fine_tuning_data = prepare_fine_tuning_data()
        
        if fine_tuning_data is None:
            raise HTTPException(status_code=500, detail="Error preparing fine-tuning data")
        
        return FineTuningDataResponse(
            total_examples=len(fine_tuning_data),
            sample_data=fine_tuning_data[:3],  # Return first 3 examples
            download_ready=True,
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preparing fine-tuning data: {str(e)}")

@app.get("/download-fine-tuning-data")
async def download_fine_tuning_data():
    """Download fine-tuning data as JSONL file"""
    try:
        fine_tuning_data = prepare_fine_tuning_data()
        
        if fine_tuning_data is None:
            raise HTTPException(status_code=500, detail="Error preparing fine-tuning data")
        
        # Convert to JSONL format
        jsonl_content = ""
        for example in fine_tuning_data:
            jsonl_content += json.dumps(example) + "\n"
        
        from fastapi.responses import Response
        
        return Response(
            content=jsonl_content,
            media_type="application/jsonl",
            headers={"Content-Disposition": "attachment; filename=faq_fine_tuning_data.jsonl"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading fine-tuning data: {str(e)}")

# Startup event
@app.on_event("startup")
async def startup_event():
    startup()

if __name__ == "__main__":
    print("Starting Advanced RAG-Powered FAQ Chatbot API...")
    print("Initializing Pinecone vector database and RAG system...")
    uvicorn.run(app, host="0.0.0.0", port=8001)