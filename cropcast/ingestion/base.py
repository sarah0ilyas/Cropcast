"""
CropCast — Base Ingester
Abstract base class for all data source connectors.
Provides retry logic, logging, schema validation, and Parquet storage.
"""

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from cropcast.config.settings import config, DATA_RAW


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=config.log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


class BaseIngester(ABC):
    """
    Abstract base for all CropCast data connectors.

    Subclasses implement:
        fetch_raw()  → raw API response
        parse()      → normalised pd.DataFrame
    """

    SOURCE_NAME: str = "unknown"

    REQUIRED_COLS = {
        "source", "country", "crop", "year",
        "metric", "value", "unit", "ingested_at",
    }

    def __init__(self):
        self.log = get_logger(self.SOURCE_NAME)
        self.session = self._build_session()

    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=2.0,
            status_forcelist=[429, 500, 502, 503, 504, 521],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({"User-Agent": "cropcast/2.0 (research)"})
        return session

    @abstractmethod
    def fetch_raw(self, **kwargs) -> Any:
        """Call the remote source and return raw payload."""

    @abstractmethod
    def parse(self, raw: Any) -> pd.DataFrame:
        """Convert raw payload to normalised DataFrame."""

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"[{self.SOURCE_NAME}] Missing columns: {missing}")
        null_rates = df[list(self.REQUIRED_COLS)].isnull().mean()
        high_null = null_rates[null_rates > 0.3]
        if not high_null.empty:
            self.log.warning("High null rates: %s", high_null.to_dict())
        return df

    def save(self, df: pd.DataFrame, partition: Optional[Dict] = None) -> Path:
        df = self.validate(df)
        df["ingested_at"] = datetime.utcnow()

        dest = DATA_RAW / self.SOURCE_NAME
        if partition:
            for k, v in partition.items():
                dest = dest / f"{k}={v}"
        dest.mkdir(parents=True, exist_ok=True)

        out = dest / "data.parquet"
        df.to_parquet(out, index=False,
                      compression=config.storage.compression,
                      engine="pyarrow")
        self.log.info("Saved %d rows → %s", len(df), out)
        return out

    def run(self, **kwargs) -> pd.DataFrame:
        self.log.info("Starting ingestion: %s", self.SOURCE_NAME)
        t0 = time.perf_counter()
        raw = self.fetch_raw(**kwargs)
        df = self.parse(raw)
        if not df.empty:
            self.save(df, partition={"year": datetime.utcnow().year})
        elapsed = time.perf_counter() - t0
        self.log.info("Finished %s — %d rows in %.1fs",
                      self.SOURCE_NAME, len(df), elapsed)
        return df

    def get(self, url: str, params: Dict = None,
            timeout: int = 30) -> requests.Response:
        self.log.debug("GET %s params=%s", url, params)
        resp = self.session.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp

    @staticmethod
    def payload_hash(data: Any) -> str:
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()
