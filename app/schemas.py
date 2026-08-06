from pydantic import BaseModel, Field


class UniverseInfo(BaseModel):
    id: str
    label: str
    n_stocks: int = Field(ge=0)
    k_options: list[int]
    description: str


class ReplicateRequest(BaseModel):
    universe: str
    k: int = Field(gt=0)


class SelectedStock(BaseModel):
    ticker: str
    sector: str
    weight: float = Field(ge=0)


class ReplicateResponse(BaseModel):
    universe: str
    k: int = Field(gt=0)
    selected: list[SelectedStock]
    cssp_objective: float = Field(ge=0)
    coverage_pct: float = Field(ge=0, le=100)
    precomputed: bool
