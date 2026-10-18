"""PyTorch MLP with an sklearn-like interface (fit / predict_proba)."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import torch
from torch import nn


class TorchMLP:
    """Small PyTorch MLP for tabular fraud scoring."""

    def __init__(
        self,
        hidden: tuple[int, ...] = (64, 32),
        epochs: int = 5,
        batch_size: int = 4096,
        lr: float = 1e-3,
        seed: int = 0,
    ) -> None:
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed

    @staticmethod
    def _prep(X: pd.DataFrame) -> np.ndarray:
        arr = X.astype("float64").copy()
        for col in arr.columns:
            if col.startswith("error_"):
                arr[col] = np.sign(arr[col]) * np.log1p(np.abs(arr[col]))
        return arr.to_numpy(dtype=np.float32)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TorchMLP:
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        Xn = self._prep(X)
        self.mean_ = Xn.mean(axis=0)
        self.std_ = Xn.std(axis=0) + 1e-6
        xt = torch.from_numpy((Xn - self.mean_) / self.std_)
        y_arr = np.asarray(y, dtype=np.float32)
        yt = torch.from_numpy(y_arr)

        layers: list[nn.Module] = []
        in_dim = xt.shape[1]
        for h in self.hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net_ = nn.Sequential(*layers)

        n_pos = max(float(y_arr.sum()), 1.0)
        n_neg = max(float(len(y_arr) - y_arr.sum()), 1.0)
        pos_weight = torch.tensor(float(np.sqrt(n_neg / n_pos)))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)

        self.net_.train()
        n = len(xt)
        for _ in range(self.epochs):
            perm = torch.from_numpy(rng.permutation(n))
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net_(xt[idx]).squeeze(1), yt[idx])
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.net_.eval()
        xt = torch.from_numpy((self._prep(X) - self.mean_) / self.std_)
        out = []
        with torch.no_grad():
            for i in range(0, len(xt), 65536):
                out.append(torch.sigmoid(self.net_(xt[i : i + 65536]).squeeze(1)).numpy())
        p = np.concatenate(out).astype(float)
        return np.column_stack([1.0 - p, p])


class RuleBaseline:
    """Rule benchmark on TRANSFER/CASH_OUT: amount "exceeds" or "equals" the old balance."""

    MODES: ClassVar[dict[str, tuple[str, ...]]] = {
        "exceeds": ("type_TRANSFER", "type_CASH_OUT", "orig_would_be_emptied"),
        "equals": ("type_TRANSFER", "type_CASH_OUT", "amount_to_orig_balance"),
    }

    def __init__(self, mode: str = "exceeds") -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}")
        self.mode = mode

    @property
    def required(self) -> tuple[str, ...]:
        return self.MODES[self.mode]

    def _check(self, X: pd.DataFrame) -> None:
        missing = [c for c in self.required if c not in X.columns]
        if missing:
            raise ValueError(f"RuleBaseline needs the 'pre' feature set; missing {missing}")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RuleBaseline:
        self._check(X)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check(X)
        if self.mode == "exceeds":
            balance_condition = X["orig_would_be_emptied"] == 1
        else:
            balance_condition = X["amount_to_orig_balance"] == 1.0
        flag = ((X["type_TRANSFER"] == 1) | (X["type_CASH_OUT"] == 1)) & balance_condition
        p = flag.to_numpy(dtype=float)
        return np.column_stack([1.0 - p, p])
