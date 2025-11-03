"""
FastAPI server implementation.
Provides REST API endpoints for the MurmurNet system.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from loguru import logger

from src.orchestrator import Orchestrator
from src.knowledge import RAGSystem, ZIMParser
from src.memory import LongTermMemory, ExperienceMemory
from src.config import config
from src.blackboard import blackboard
from src.utils.agent_tracking import agent_tracker
from src.utils.metrics import performance_metrics


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

BASE_DIR = Path(__file__).resolve().parents[2]
UI_STATIC_DIR = BASE_DIR / "webui" / "static"

if UI_STATIC_DIR.exists():
    app.mount(
        "/ui/assets",
        StaticFiles(directory=str(UI_STATIC_DIR)),
        name="ui-static",
    )


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
        "ui": "/ui",
    }


@app.get("/ui", include_in_schema=False)
async def ui_index():
    """Serve the monitoring UI."""
    if not UI_STATIC_DIR.exists():
        raise HTTPException(status_code=404, detail="UI assets not available")
    return FileResponse(UI_STATIC_DIR / "index.html")


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


@app.get("/ui/tasks", tags=["UI"])
async def ui_tasks():
    """Get summaries for dashboard task list."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    summaries = await blackboard.get_all_task_summaries()
    activity_map = await agent_tracker.get_overview()

    tasks = []
    for summary in summaries:
        task_id = summary["task_id"]
        tasks.append({
            **summary,
            "activity": activity_map.get(task_id, {}),
        })

    return {"tasks": tasks}


@app.get("/ui/task/{task_id}/blackboard", tags=["UI"])
async def ui_task_blackboard(task_id: str, limit: int = 50):
    """Get recent blackboard entries for a task."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    entries = await blackboard.read_entries(task_id=task_id, limit=limit)
    if not entries:
        summary = await blackboard.get_task_summary(task_id)
        if summary is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    data = []
    for entry in entries[-limit:]:
        preview = str(entry.content)
        if len(preview) > 400:
            preview = preview[:397] + "..."
        data.append(
            {
                "entry_id": entry.entry_id,
                "timestamp": entry.timestamp.isoformat(),
                "agent_id": entry.agent_id,
                "entry_type": entry.entry_type.value,
                "content_preview": preview,
                "metadata": entry.metadata.dict(),
            }
        )

    return {"task_id": task_id, "entries": data}


@app.get("/ui/metrics", tags=["UI"])
async def ui_metrics():
    """Get aggregate performance metrics for dashboard."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    performance = await performance_metrics.get_summary()
    agent_stats = await agent_tracker.get_agent_stats()
    system_stats = orchestrator.get_statistics()

    return {
        "performance": performance,
        "agent_stats": agent_stats,
        "system": system_stats,
    }


@app.post("/ui/query", tags=["UI"])
async def ui_query(request: QueryRequest):
    """Start a query asynchronously for the dashboard playground."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    task_id = await orchestrator.start_query(request.query)
    return {"task_id": task_id, "status": "running"}


@app.get("/ui/task/{task_id}/result", tags=["UI"])
async def ui_task_result(task_id: str):
    """Fetch the latest result for a task if available."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    result = await agent_tracker.get_result(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not ready")
    return result


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
