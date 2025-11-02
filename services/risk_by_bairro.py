# -*- coding: utf-8 -*-
"""
Une H_score (cidade) + U_t (bairros) -> Risk_score por bairro/data.
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
import os
import yaml, json
import sys

CITY = "canoas"

# Define a base path dynamically (project root) to ensure the script finds resources
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Altere conforme sua estrutura
SCRIPT_DIR = Path(__file__).parent.resolve()  # Diretório atual do script
DATA = PROJECT_ROOT / "giovanni-algoritmo"
HAZARD_CSV = DATA / "services" / "data" / "hazard" / "hazard_forecast.csv"
U_CSV = DATA / "services" / "data" / "u" / "canoas_bairros_u.csv"
U_GEOJSON = DATA / "services" / "data" / "u" / "canoas_bairros_u.geojson"
OUT_DIR = DATA / "services" / "data" / "risk"
OUT_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_YAML = DATA / "configs" / "weights.yaml"


def main():
    # Ensure all required files exist
    for file in [HAZARD_CSV, U_CSV, U_GEOJSON, WEIGHTS_YAML]:
        if not file.exists():
            sys.exit(f"❌ Required file not found: {file}")

    # Load weights and thresholds
    w = yaml.safe_load(WEIGHTS_YAML.read_text())
    green, yellow = w["hazard_levels"]["green_max"], w["hazard_levels"]["yellow_max"]

    # 1) Read H_score (cidade)
    try:
        dfH = pd.read_csv(HAZARD_CSV, parse_dates=["date"])
    except Exception as e:
        sys.exit(f"❌ Failed to read HAZARD_CSV ({HAZARD_CSV}): {e}")
    if "H_score" not in dfH.columns:
        sys.exit("❌ Column 'H_score' missing in hazard_forecast.csv")
    dfH = dfH[["date", "H_score"]]

    # 2) Read U data by bairro
    try:
        dfU = pd.read_csv(U_CSV)
        gdfU = gpd.read_file(U_GEOJSON)
    except Exception as e:
        sys.exit(f"❌ Failed to read U_CSV or U_GEOJSON: {e}")
    if "U_t" not in dfU.columns:
        sys.exit("❌ Column 'U_t' missing in U CSV")
    dfU["Fragilidade_t"] = 1 - dfU["U_t"]

    # 3) Cross-join (assign same H_score to all bairros by date)
    dfH["key"] = 1
    dfU["key"] = 1
    dfR = dfH.merge(dfU, on="key", suffixes=("_hazard", "_infra"))
    dfR.drop(columns="key", inplace=True)

    # 4) Calculate Risk
    dfR["Risk_score"] = (dfR["H_score"] * dfR["Fragilidade_t"]).clip(0, 1)

    def bucket(x):
        if x < green: return "green"
        if x < yellow: return "yellow"
        return "red"

    dfR["Risk_level"] = dfR["Risk_score"].apply(bucket)

    # 5) Aggregate and export (CSV and GeoJSON)
    out_csv = OUT_DIR / f"{CITY}_bairros_risk.csv"
    dfR_out = dfR[[
        "bairro", "date", "H_score", "U_t", "Fragilidade_t", "Risk_score", "Risk_level"
    ]]
    dfR_out.to_csv(out_csv, index=False)

    # GeoJSON -> export last day's data
    last_date = dfR["date"].max()
    gdf_last = gdfU.merge(
        dfR[dfR["date"] == last_date][["bairro", "Risk_score", "Risk_level"]],
        on="bairro",
        how="left"
    )
    gdf_last.to_file(OUT_DIR / f"{CITY}_bairros_risk.geojson", driver="GeoJSON")

    print(f"✅ Risk generated: {len(dfR_out)} rows, {len(gdf_last)} bairros.")
    print(f"- CSV: {out_csv}")
    print(f"- GeoJSON: {OUT_DIR / f'{CITY}_bairros_risk.geojson'}")


if __name__ == "__main__":
    main()
