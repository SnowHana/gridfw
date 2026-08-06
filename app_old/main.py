import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app_old import universe_registry
from app_old.schemas import ReplicateRequest, ReplicateResponse, UniverseInfo

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(title="Sparse Index Replication")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/universes", response_model=list[UniverseInfo])
def universes():
    return universe_registry.list_universes()


@app.post("/replicate", response_model=ReplicateResponse)
def replicate(req: ReplicateRequest):
    try:
        return universe_registry.get_replication(req.universe, req.k)
    except universe_registry.UnknownUniverseError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except universe_registry.UnsupportedKError as e:
        raise HTTPException(status_code=422, detail=str(e))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
