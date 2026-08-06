from pydantic import BaseModel, Field


class UniverseInfo(BaseModel):
    id: str
    label: str
    n_stocks: int
    k_options: list[int]
    description: str


class ReplicateRequest(BaseModel):
    universe: str
    k: int = Field(gt=0)


class SelectedStock(BaseModel):
    ticker: str
    sector: str
    weight: float


class ReplicateResponse(BaseModel):
    universe: str
    k: int
    selected: list[SelectedStock]
    cssp_objective: float
    coverage_pct: float
    precomputed: bool
