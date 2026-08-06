from fastapi import FastAPI

app = FastAPI(title="Sparse Index Replication")


@app.get("/health")
def health():
    return {"status": "ok"}
