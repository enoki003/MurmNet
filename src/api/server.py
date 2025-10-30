"""
FastAPI server implementation.
Provides REST API endpoints for the MurmurNet system.
"""

from datetime import datetime
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

from src.orchestrator import Orchestrator
from src.knowledge import RAGSystem, ZIMParser
from src.memory import LongTermMemory, ExperienceMemory
from src.config import config


# Request/Response models
class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., description="User query", min_length=1)
    task_id: Optional[str] = Field(default=None, description="Optional custom task ID")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    task_id: str
    query: str
    answer: Optional[str] = None
    error: Optional[str] = None
    execution_time_seconds: float
    success: bool
    timestamp: str


class HealthResponse(BaseModel):
    """Response model for health endpoint."""
    status: str
    system_name: str
    timestamp: str
    statistics: Dict


class TaskHistoryResponse(BaseModel):
    """Response model for task history endpoint."""
    task_id: str
    history: Dict


# Initialize FastAPI app
app = FastAPI(
    title="MurmurNet API",
    description="API for Small Language Model Swarm with Emergent Intelligence",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instance
orchestrator: Optional[Orchestrator] = None


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    global orchestrator
    
    logger.info("Starting MurmurNet API server...")
    
    # Ensure directories exist
    config.ensure_directories()
    
    # Initialize RAG system if ZIM file is available
    rag_system = None
    if config.knowledge_base.zim_file_path and config.knowledge_base.zim_file_path.exists():
        try:
            logger.info("Initializing RAG system...")
            zim_parser = ZIMParser()
            rag_system = RAGSystem(zim_parser=zim_parser)
            
            # Try to load existing database
            try:
                rag_system.load_database()
                logger.info("Loaded existing knowledge base")
            except:
                logger.info("No existing knowledge base found")
        except Exception as e:
            logger.warning(f"Failed to initialize RAG system: {e}")
    else:
        logger.warning("No ZIM file configured, RAG system disabled")
    
    # Initialize memory systems
    long_term_memory = LongTermMemory()
    experience_memory = ExperienceMemory()
    
    # Initialize orchestrator
    orchestrator = Orchestrator(
        rag_system=rag_system,
        long_term_memory=long_term_memory,
        experience_memory=experience_memory,
    )
    
    logger.info("MurmurNet API server started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down MurmurNet API server...")
    
    if orchestrator and orchestrator.llm:
        orchestrator.llm.unload()
    
    logger.info("Shutdown complete")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to MurmurNet API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    stats = orchestrator.get_statistics()
    
    return HealthResponse(
        status="healthy",
        system_name=config.system.system_name,
        timestamp=datetime.utcnow().isoformat(),
        statistics=stats,
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def process_query(request: QueryRequest):
    """
    Process a user query through the agent swarm.
    
    Args:
        request: Query request
        
    Returns:
        Query response with answer or error
    """
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        result = await orchestrator.process_query(request.query)
        
        return QueryResponse(
            task_id=result["task_id"],
            query=result["query"],
            answer=result.get("answer"),
            error=result.get("error"),
            execution_time_seconds=result["execution_time_seconds"],
            success=result["success"],
            timestamp=datetime.utcnow().isoformat(),
        )
        
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/task/{task_id}/history", response_model=TaskHistoryResponse, tags=["Task"])
async def get_task_history(task_id: str):
    """
    Get complete history of a task.
    
    Args:
        task_id: Task identifier
        
    Returns:
        Task history
    """
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        history = await orchestrator.get_task_history(task_id)
        
        if history is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        return TaskHistoryResponse(
            task_id=task_id,
            history=history,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving task history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/statistics", tags=["Statistics"])
async def get_statistics():
    """Get system statistics."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return orchestrator.get_statistics()


@app.post("/knowledge/index", tags=["Knowledge"])
async def index_knowledge(
    max_articles: Optional[int] = None,
    background_tasks: BackgroundTasks = None,
):
    """
    Index documents from ZIM file into knowledge base.
    
    Args:
        max_articles: Maximum number of articles to index
        background_tasks: FastAPI background tasks
        
    Returns:
        Status message
    """
    if orchestrator is None or orchestrator.rag_system is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not available"
        )
    
    # Run indexing in background
    def index_task():
        try:
            count = orchestrator.rag_system.index_documents(max_articles=max_articles)
            logger.info(f"Indexed {count} document chunks")
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
    
    if background_tasks:
        background_tasks.add_task(index_task)
        return {"message": "Indexing started in background", "max_articles": max_articles}
    else:
        index_task()
        return {"message": "Indexing completed", "max_articles": max_articles}


# For development
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.server:app",
        host=config.api.api_host,
        port=config.api.api_port,
        reload=False,
        log_level=config.system.log_level.value.lower(),
    )
