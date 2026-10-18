"""FastAPI scoring service; loads the model bundle from MODEL_PATH at startup."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .config import MODEL_PATH
from .serving import load_bundle, score_transaction


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.environ.get("MODEL_PATH", MODEL_PATH))
    app.state.bundle = load_bundle(path) if path.exists() else None
    yield


app = FastAPI(title="txn-intelligence", version=__version__, lifespan=lifespan)
app.state.bundle = None


class Transaction(BaseModel):
    """Pre-approval fields only."""

    step: int = Field(ge=0, description="Hour index; hour of day is step % 24")
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    amount: float = Field(ge=0)
    old_balance_orig: float = Field(ge=0, description="Sender balance before the transaction")
    old_balance_dest: float = Field(ge=0, description="Receiver balance before the transaction")


class Score(BaseModel):
    """The fraud score is a ranking score, not a calibrated probability."""

    fraud_score: float
    flagged: bool
    threshold: float
    rule_flag: bool
    model_version: str


@app.get("/health")
def health() -> dict:
    bundle = app.state.bundle
    return {
        "status": "ok",
        "model_loaded": bundle is not None,
        "model_version": bundle.version if bundle else None,
    }


@app.post("/score", response_model=Score)
def score(txn: Transaction) -> Score:
    bundle = app.state.bundle
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded; train it first")
    return Score(**score_transaction(bundle, txn.model_dump()))
