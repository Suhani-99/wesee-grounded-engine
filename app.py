"""
app.py — the HTTP service. Exposes POST /ask.

This is a THIN wrapper: all the grounding logic lives in engine.py. Here we
just (1) build the index once at startup, (2) accept a question over HTTP,
(3) hand it to engine.answer(), (4) return the JSON the spec defines.

Run (single command):  python app.py
Then POST to:           http://localhost:8000/ask
"""

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os


from engine import build_index, answer

app = FastAPI(title="WeSee Grounded Answer Engine")

# --- Build the index ONCE at startup ---
# WHY: embedding the corpus costs a few seconds. We pay that cost once when the
# server boots, then reuse the index for every request. Rebuilding per request
# would make every call slow for no reason. <-- Q&A: startup cost paid once;
# each request is just retrieve + generate.
print("Building index at startup (embedding the corpus)...")
INDEX = build_index()
print(f"Ready. Indexed {len(INDEX)} chunks.")


class AskRequest(BaseModel):
    """
    The request body shape: {"question": "..."}.
    WHY pydantic: FastAPI uses this to validate incoming JSON automatically —
    if 'question' is missing or the wrong type, the caller gets a clear 422
    error instead of the server crashing.
    """
    question: str


@app.post("/ask")
def ask(req: AskRequest):
    """
    POST /ask -> grounded answer with citations, or an honest refusal.

    Returns exactly the spec's shape:
      {"answer": str, "citations": [{"doc","quote"}], "answered": bool}

    All the real work (retrieval, grounded generation, citation verification,
    injection resistance) happens inside engine.answer(). This handler just
    passes the question through and returns the result.
    """
    result = answer(req.question, INDEX)
    return result


@app.get("/")
def health():
    """A simple health check so you can confirm the server is up in a browser."""
    return {"status": "ok", "service": "WeSee Grounded Answer Engine"}


# --- Start the server when run directly ---
# WHY here: lets the whole app launch with 'python app.py' — one command, as
# the brief requires. uvicorn is the ASGI server that actually runs FastAPI.
if __name__ == "__main__":
    # Use the host's provided port if present (deploy platforms set $PORT),
    # else default to 8000 for local runs.
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)