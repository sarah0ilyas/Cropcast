"""
CropCast — Weather Ingester
Pulls historical daily weather from Open-Meteo for all
configured production regions. Free, no API key required.
"""

import time
from datetime import datetime, date
from typing import Any, Dict

import pandas as pd

from cropcast.config.settings import config
from cropcast.ingestion.base import BaseIngester


class WeatherIngester(BaseIngester):
    SOURCE_NAME = "open_meteo_weather"

    UNIT_MAP = {
        "temperature_2m_max":         "°C",
        "temperature_2m_min":         "°C",
        "precipitation_sum":          "mm",
        "et0_fao_evapotranspiration": "mm",
    }

    COUNTRY_MAP = {
        "Ica_Peru":            "Peru",
        "Maule_Chile":         "Chile",
        "Western_Cape_SA":     "South Africa",
        "Murcia_Spain":        "Spain",
        "California_US":       "United States of America",
        "Mendoza_Argentina":   "Argentina",
        "Southeast_Australia": "Australia",
        "Bordeaux_France":     "France",
        "Aegean_Turkey":       "Turkey",
        "Maharashtra_India":   "India",
        "Alentejo_Portugal":   "Portugal",
        "Crete_Greece":        "Greece",
    }

    def __init__(self):
        super().__init__()
        self.cfg = config.weather

    def fetch_raw(self, **kwargs) -> Dict[str, Any]:
        results = {}
        for location, (lat, lon) in self.cfg.locations.items():
            self.log.info("Fetching weather: %s (%.2f, %.2f)", location, lat, lon)
            params = {
                "latitude":   lat,
                "longitude":  lon,
                "start_date": self.cfg.start_date,
                "end_date":   date.today().isoformat(),
                "daily":      ",".join(self.cfg.variables),
                "timezone":   "UTC",
            }
            try:
                resp = self.get(self.cfg.base_url, params=params,
                                timeout=self.cfg.timeout)
                results[location] = resp.json()
                n = len(resp.json().get("daily", {}).get("time", []))
                self.log.info("  -> %d days for %s", n, location)
            except Exception as exc:
                self.log.error("Weather failed for %s: %s", location, exc)
            time.sleep(5)
        return results

    def parse(self, raw: Dict[str, Any]) -> pd.DataFrame:
        dfs = []
        for location, payload in raw.items():
            daily = payload.get("daily", {})
            if not daily or "time" not in daily:
                continue

            df = pd.DataFrame({"date": pd.to_datetime(daily["time"])})
            for var in self.cfg.variables:
                if var in daily:
                    df[var] = daily[var]

            lat = payload.get("latitude", 0)
            df["month"] = df["date"].dt.month
            df["in_growing_season"] = (
                df["month"].isin([10, 11, 12, 1, 2, 3, 4])
                if lat < 0
                else df["month"].isin([4, 5, 6, 7, 8, 9, 10])
            )

            value_vars = [v for v in self.cfg.variables if v in df.columns]
            df_long = df.melt(
                id_vars=["date", "month", "in_growing_season"],
                value_vars=value_vars,
                var_name="metric", value_name="value"
            ).dropna(subset=["value"])

            df_long["source"]      = self.SOURCE_NAME
            df_long["location"]    = location
            df_long["lat"]         = payload.get("latitude")
            df_long["lon"]         = payload.get("longitude")
            df_long["year"]        = df_long["date"].dt.year
            df_long["country"]     = self.COUNTRY_MAP.get(location, location)
            df_long["crop"]        = "all"
            df_long["unit"]        = df_long["metric"].map(self.UNIT_MAP)
            df_long["ingested_at"] = datetime.utcnow()

            dfs.append(df_long)

        if not dfs:
            return pd.DataFrame()

        out = pd.concat(dfs, ignore_index=True)
        self.log.info("Parsed %d weather rows across %d locations",
                      len(out), len(raw))
        return out
