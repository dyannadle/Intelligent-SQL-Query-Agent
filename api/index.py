import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from agent.graph import build_graph

load_dotenv()

app = FastAPI(title="Intelligent SQL Query Agent")

# Initialize graph only once at startup
sql_agent = build_graph()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    query_executed: str = ""

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Intelligent SQL Query Agent is running! Send a POST request to /api/chat with your question.",
        "docs": "/docs"
    }

@app.post("/api/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    if not os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY") == "your_groq_api_key_here":
         raise HTTPException(status_code=500, detail="Missing GROQ_API_KEY.")

    initial_state = {
        "question": request.question,
        "sql_query": "",
        "is_valid_sql": False,
        "validation_error": "",
        "query_result": "",
        "final_answer": "",
        "error_count": 0
    }
    
    try:
        final_output = sql_agent.invoke(initial_state)
        
        if final_output.get("final_answer"):
             return QueryResponse(
                 answer=final_output["final_answer"], 
                 query_executed=final_output.get("sql_query", "")
             )
        elif final_output.get("error_count", 0) >= 3:
             return QueryResponse(answer="Agent stopped after multiple failed attempts to generate or execute valid SQL.")
        else:
             return QueryResponse(answer="Agent could not generate an answer.")
             
    except Exception as e:
         error_msg = str(e)
         if "rate_limit_exceeded" in error_msg or "429" in error_msg:
             return QueryResponse(answer=f"Groq Free Tier Rate Limit Exceeded. Please wait a minute and try again.\\n\\nDetails: {error_msg}")
             
         raise HTTPException(status_code=500, detail=error_msg)
