import os

if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # torch + faiss OpenMP on Windows
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
