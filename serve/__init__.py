import os

if os.name == "nt":
    # torch + faiss both ship an OpenMP runtime; on Windows the second one aborts without this
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# One compute thread per request. Concurrency comes from many requests (and many worker processes),
# not from each request spawning a pool of threads — otherwise 50 parallel requests x N OpenMP threads
# fight over the cores and p95 explodes. Must be set before numpy / torch / lightgbm are imported.
_threads = os.environ.get("SERVE_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, _threads)
