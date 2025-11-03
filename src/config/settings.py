"""
Configuration management for MurmurNet system.
Provides centralized configuration loading and validation.
"""

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(str, Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ModelDevice(str, Enum):
    """Model device enumeration."""
    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"


class VectorDBType(str, Enum):
    """Vector database type enumeration."""
    FAISS = "faiss"
    CHROMA = "chroma"


class ModelQuantization(str, Enum):
    """Model quantization enumeration."""
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    Q4 = "q4"


class SystemConfig(BaseSettings):
    """System-wide configuration."""
    
    system_name: str = Field(default="MurmurNet")
    log_level: LogLevel = Field(default=LogLevel.INFO)
    log_dir: Path = Field(default=Path("./logs"))
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class ModelConfig(BaseSettings):
    """Model configuration."""
    
    default_model_name: str = Field(default="meta-llama/Llama-3.2-3B-Instruct")
    model_cache_dir: Path = Field(default=Path("./models"))
    model_device: ModelDevice = Field(default=ModelDevice.CUDA)
    model_quantization: ModelQuantization = Field(default=ModelQuantization.Q4)
    
    # Embedding model configuration
    embedding_model: str = Field(default="intfloat/multilingual-e5-large")
    embedding_device: ModelDevice = Field(default=ModelDevice.CUDA)
    embedding_batch_size: int = Field(default=32)
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class VectorDBConfig(BaseSettings):
    """Vector database configuration."""
    
    vector_db_type: VectorDBType = Field(default=VectorDBType.FAISS)
    vector_db_path: Path = Field(default=Path("./data/vector_db"))
    vector_db_dimension: int = Field(default=1024)
    vector_db_top_k: int = Field(default=5)
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class MemoryConfig(BaseSettings):
    """Memory system configuration."""
    
    memory_db_path: Path = Field(default=Path("./data/memory_db"))
    long_term_memory_top_k: int = Field(default=3)
    experience_memory_top_k: int = Field(default=3)
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class KnowledgeBaseConfig(BaseSettings):
    """Knowledge base configuration."""
    
    zim_file_path: Optional[Path] = Field(default=None)
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=50)
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class BlackboardConfig(BaseSettings):
    """Blackboard system configuration."""
    
    blackboard_max_entries: int = Field(default=1000)
    blackboard_retention_hours: int = Field(default=24)
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class APIConfig(BaseSettings):
    """API server configuration."""
    
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=1)
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class AgentConfig(BaseSettings):
    """Agent system configuration."""
    
    max_agent_retries: int = Field(default=3)
    agent_timeout_seconds: int = Field(default=60)
    max_parallel_agents: int = Field(default=3)
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class EvaluationConfig(BaseSettings):
    """Evaluation configuration."""
    
    evaluation_output_dir: Path = Field(default=Path("./evaluation/results"))
    benchmark_data_dir: Path = Field(default=Path("./evaluation/benchmarks"))
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class Config:
    """
    Main configuration container.
    Aggregates all configuration sections.
    """
    
    def __init__(self):
        self.system = SystemConfig()
        self.model = ModelConfig()
        self.vector_db = VectorDBConfig()
        self.memory = MemoryConfig()
        self.knowledge_base = KnowledgeBaseConfig()
        self.blackboard = BlackboardConfig()
        self.api = APIConfig()
        self.agent = AgentConfig()
        self.evaluation = EvaluationConfig()
        
    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        directories = [
            self.system.log_dir,
            self.model.model_cache_dir,
            self.vector_db.vector_db_path,
            self.memory.memory_db_path,
            self.evaluation.evaluation_output_dir,
            self.evaluation.benchmark_data_dir,
        ]
        
        if self.knowledge_base.zim_file_path:
            directories.append(self.knowledge_base.zim_file_path.parent)
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = Config()
