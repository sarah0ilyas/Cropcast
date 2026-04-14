"""
CropCast — FAO Ingester
Reads FAO bulk CSV and filters to configured crops and countries.
Uses bulk CSV as primary source (more reliable than API).

Download bulk CSV from:
https://fenixservices.fao.org/faostat/static/bulkdownloads/Production_Crops_Livestock_E_All_Data_(Normalized).zip
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from cropcast.config.settings import config
from cropcast.ingestion.base import BaseIngester


class FAOIngester(BaseIngester):
    SOURCE_NAME = "faostat"

    ELEMENT_MAP = {
        "Area harvested": ("area_ha",       "HA",     1000.0),
        "Production":     ("production_mt", "MT",     1000.0),
        "Yield":          ("yield_mt_ha",   "MT/HA",  1/10000),
    }

    def __init__(self, csv_path: str = None):
        super().__init__()
        self.cfg = config.fao
        self.csv_path = Path(csv_path) if csv_path else None

    def fetch_raw(self, **kwargs) -> pd.DataFrame:
        csv_path = kwargs.get("csv_path") or self.csv_path
        if not csv_path or not Path(csv_path).exists():
            raise FileNotFoundError(
                f"FAO bulk CSV not found at: {csv_path}\n"
                "Download from: https://fenixservices.fao.org/faostat/static/"
                "bulkdownloads/Production_Crops_Livestock_E_All_Data_(Normalized).zip"
            )

        self.log.info("Reading FAO bulk CSV: %s", csv_path)
        self.log.info("Filtering: %d crops × %d countries",
                      len(self.cfg.crops), len(self.cfg.area_codes))

        chunks = []
        for chunk in pd.read_csv(csv_path, chunksize=100_000,
                                 encoding="latin-1", low_memory=False):
            filtered = chunk[
                chunk["Item"].isin(self.cfg.crops) &
                chunk["Area"].isin(self.cfg.area_codes) &
                chunk["Element"].isin(self.ELEMENT_MAP.keys()) &
                (chunk["Year"] >= self.cfg.start_year)
            ]
            if len(filtered) > 0:
                chunks.append(filtered)

        if not chunks:
            self.log.warning("No matching rows found in FAO CSV")
            return pd.DataFrame()

        result = pd.concat(chunks, ignore_index=True)
        self.log.info("Found %d matching rows", len(result))
        return result

    def parse(self, raw: pd.DataFrame) -> pd.DataFrame:
        if raw.empty:
            return pd.DataFrame()

        rows = []
        for _, rec in raw.iterrows():
            element = rec.get("Element", "")
            if element not in self.ELEMENT_MAP:
                continue
            metric, unit, conversion = self.ELEMENT_MAP[element]
            try:
                value = float(rec["Value"]) * conversion
            except (TypeError, ValueError):
                continue

            rows.append({
                "source":      self.SOURCE_NAME,
                "country":     rec["Area"],
                "crop":        rec["Item"],
                "year":        int(rec["Year"]),
                "metric":      metric,
                "value":       round(value, 4),
                "unit":        unit,
                "flag":        rec.get("Flag", ""),
                "ingested_at": datetime.utcnow(),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df = df.drop_duplicates(subset=["country", "crop", "year", "metric"])
        self.log.info("Parsed %d clean FAO rows", len(df))
        return df

    def run(self, csv_path: str = None, **kwargs) -> pd.DataFrame:
        raw = self.fetch_raw(csv_path=csv_path or self.csv_path)
        df = self.parse(raw)
        if not df.empty:
            self.save(df, partition={"year": datetime.utcnow().year})
        return df
