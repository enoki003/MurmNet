#!/usr/bin/env python
"""Check environment setup"""

import sys

print("=== Environment Check ===\n")

# Check Python version
print(f"Python: {sys.version}")

# Check torch
try:
    import torch
    print(f"✓ torch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
except ImportError as e:
    print(f"✗ torch: {e}")

# Check transformers
try:
    import transformers
    print(f"✓ transformers: {transformers.__version__}")
except ImportError as e:
    print(f"✗ transformers: {e}")

# Check faiss
try:
    import faiss
    version = faiss.__version__ if hasattr(faiss, '__version__') else 'OK'
    print(f"✓ faiss: {version}")
except ImportError as e:
    print(f"✗ faiss: {e}")

# Check sentence-transformers
try:
    import sentence_transformers
    print(f"✓ sentence-transformers: {sentence_transformers.__version__}")
except ImportError as e:
    print(f"✗ sentence-transformers: {e}")

# Check fastapi
try:
    import fastapi
    print(f"✓ fastapi: {fastapi.__version__}")
except ImportError as e:
    print(f"✗ fastapi: {e}")

# Check other dependencies
for lib in ['pydantic', 'loguru', 'tenacity', 'chromadb', 'beautifulsoup4']:
    try:
        mod = __import__(lib.replace('-', '_'))
        version = mod.__version__ if hasattr(mod, '__version__') else 'OK'
        print(f"✓ {lib}: {version}")
    except ImportError as e:
        print(f"✗ {lib}: {e}")

print("\n=== Environment check complete ===")
