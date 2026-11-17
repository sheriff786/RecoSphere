"""Pydantic request/response models ("schemas"): what a caller may send and what it gets back. Services take
and return these, never raw dicts; routers (phase D) use them as FastAPI bodies and response models."""
