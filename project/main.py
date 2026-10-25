from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent import run_agent, CHART_DIR

app = FastAPI(title="Stock Analysis Agent API")

# Allow a browser frontend (React, plain HTML, etc.) to call this API.
# Replace "*" with your frontend URL when you deploy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chart images saved by the agent are served at /charts/<file>.png
app.mount("/charts", StaticFiles(directory=CHART_DIR), name="charts")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, examples=["Show graph and 7-day prediction for RELIANCE.NS"])
    session_id: str | None = None  # same id = the agent remembers earlier messages


class ChatResponse(BaseModel):
    reply: str
    chart_urls: list[str] = []


# Conversation memory kept in RAM: {session_id: [{"role", "content"}, ...]}
# Lost on server restart. Use a database/Redis if you need it to persist.
SESSIONS: dict[str, list[dict]] = {}
MAX_SESSIONS = 200
MAX_HISTORY = 10  # last 10 messages (5 exchanges)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse("index.html")  # keep index.html next to main.py


@app.get("/health")
def health():
    return {"status": "ok"}


# Plain `def` (not `async def`): the agent is blocking and slow (LSTM training),
# so FastAPI runs it in a worker thread instead of freezing the event loop.
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        history = SESSIONS.get(req.session_id, []) if req.session_id else []
        out = run_agent(req.message.strip(), history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if req.session_id:
        if len(SESSIONS) >= MAX_SESSIONS and req.session_id not in SESSIONS:
            SESSIONS.pop(next(iter(SESSIONS)))  # drop the oldest session
        SESSIONS[req.session_id] = (history + [
            {"role": "user", "content": req.message.strip()},
            {"role": "assistant", "content": out["reply"]},
        ])[-MAX_HISTORY:]

    # "charts/AAPL_ab12cd34.png" -> "/charts/AAPL_ab12cd34.png"
    urls = ["/" + p for p in out["charts"]]
    return ChatResponse(reply=out["reply"], chart_urls=urls)