# -*- coding: utf-8 -*-
"""
FastAPI - Risco por Bairros (Canoas)
------------------------------------
- Combina Perigo (H) e Robustez (U) para gerar Risco por bairro/data.
- Trata U==0 ou NaN como "no_data" (cinza; fora de ranking).
- Inclui geração de INSIGHTS via OpenAI (RAG simples com dados locais).

Endpoints principais:
- /health
- /v1/meta
- /v1/risk/by_bairro            (JSON tabular com filtros)
- /v1/risk/by_bairro/csv        (CSV)
- /v1/risk/by_bairro/top        (Top-N por data)
- /v1/geo/canoas/bairros_risk   (GeoJSON mapa por data, com include=basic|infra|hazard|all)
- /v1/bairros/detail            (Detalhe de um bairro em uma data; U dinâmico opcional)
- /v1/filters                   (Esquema de filtros)
- /v1/insights/by_bairro        (Narrativa + ações por bairro/data via OpenAI)
- /v1/insights/city_top         (Síntese municipal top-N por data via OpenAI)

Requisitos de arquivo:
- data/hazard/hazard_forecast.csv       (date, H_score, [p6_pct,a72_pct,sm_norm,et_deficit,p1_pct,pp_unit,rd_norm])
- data/u/canoas_bairros_u.csv           (por bairro: U_t/U_static + sub-índices + métricas)
- data/u/canoas_bairros_u.geojson       (geometria + as mesmas propriedades)
- data/pop/canoas_bairros_pop.csv       (opcional: bairro,population)
"""

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from typing import Optional, Dict, Any
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import yaml, json, os, textwrap
from datetime import date, timedelta, timezone

# OpenAI (insights)
try:
    from openai import OpenAI
    OPENAI_SDK_OK = True
except Exception:
    OPENAI_SDK_OK = False

# Open-Meteo (apenas p/ recálculo dinâmico de U no detalhe, se desejar)
try:
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry
    OM_AVAILABLE = True
except Exception:
    OM_AVAILABLE = False

# ----------------------------------- Paths -----------------------------------

ROOT = Path(".").resolve()
DATA = ROOT / "services" / "data"
HAZARD_CSV = DATA / "hazard" / "hazard_forecast.csv"
U_CSV = DATA / "u" / "canoas_bairros_u.csv"
U_GEOJSON = DATA / "u" / "canoas_bairros_u.geojson"
POP_CSV = DATA / "pop" / "canoas_bairros_pop.csv"   # opcional
WEIGHTS_YAML = ROOT / "configs" / "weights.yaml"

# ----------------------------------- App -------------------------------------

app = FastAPI(title="Canoas - Risco por Bairros API", version="1.2.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ----------------------------------- Utils -----------------------------------

def load_weights() -> Dict[str, Any]:
    if WEIGHTS_YAML.exists():
        return yaml.safe_load(WEIGHTS_YAML.read_text(encoding="utf-8"))
    return {
        "hazard_levels": {"green_max": 0.33, "yellow_max": 0.66},
        "hazard_daily_weights": {"p6":0.25,"a72":0.25,"sm":0.15,"etd":0.10,"p1":0.10,"pp":0.05,"rd":0.10},
        "u_weights": {"perm":0.40,"macro":0.25,"cob":0.20,"micro":0.15},
    }

def try_load_hazard() -> pd.DataFrame:
    if not HAZARD_CSV.exists():
        raise HTTPException(404, detail="hazard_forecast.csv não encontrado em data/hazard/")
    df = pd.read_csv(HAZARD_CSV)
    if "date" not in df.columns or "H_score" not in df.columns:
        raise HTTPException(500, detail="hazard_forecast.csv deve conter colunas 'date' e 'H_score'.")
    df["date"] = pd.to_datetime(df["date"])
    return df

def try_load_u() -> (pd.DataFrame, gpd.GeoDataFrame):
    if not U_CSV.exists() or not U_GEOJSON.exists():
        raise HTTPException(404, detail="Arquivos de U não encontrados (CSV/GeoJSON).")
    dfU = pd.read_csv(U_CSV)
    gdfU = gpd.read_file(U_GEOJSON)
    if gdfU.crs is None:
        gdfU.set_crs(epsg=4326, inplace=True)
    else:
        gdfU = gdfU.to_crs(epsg=4326)
    # Campo de nome do bairro
    name_field = None
    for cand in ["bairro","name","NOME","BAIRRO","Bairro"]:
        if cand in gdfU.columns:
            name_field = cand
            break
    if name_field is None:
        name_field = "bairro"
        gdfU[name_field] = gdfU["OBJECTID"].astype(str)
    # Harmoniza "bairro"
    if "bairro" not in dfU.columns and name_field in dfU.columns:
        dfU.rename(columns={name_field:"bairro"}, inplace=True)
    if "bairro" not in gdfU.columns and name_field in gdfU.columns:
        gdfU.rename(columns={name_field:"bairro"}, inplace=True)
    dfU["bairro"] = dfU["bairro"].astype(str)
    gdfU["bairro"] = gdfU["bairro"].astype(str)
    return dfU, gdfU

def bucket_risk(x: float, thresholds: Dict[str,float]) -> str:
    if pd.isna(x): return "no_data"
    g = thresholds["green_max"]; y = thresholds["yellow_max"]
    if x < g: return "green"
    if x < y: return "yellow"
    return "red"

def apply_filters(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    # Risk level
    levels = params.get("risk_level")
    if levels:
        allowed = set([s.strip().lower() for s in levels.split(",")])
        out = out[out["Risk_level"].str.lower().isin(allowed)]
    # Score range
    min_r = params.get("min_risk"); max_r = params.get("max_risk")
    if min_r is not None: out = out[out["Risk_score"] >= float(min_r)]
    if max_r is not None: out = out[out["Risk_score"] <= float(max_r)]
    # Hazard factors (se existirem no hazard CSV)
    for k in ["p6_pct","a72_pct","sm_norm","et_deficit","p1_pct","pp_unit","rd_norm"]:
        lo = params.get(f"min_{k}"); hi = params.get(f"max_{k}")
        if lo is not None and k in out.columns: out = out[out[k] >= float(lo)]
        if hi is not None and k in out.columns: out = out[out[k] <= float(hi)]
    # Infra subíndices
    for k in ["u_cobertura","u_micro","u_macro","u_permeabilidade"]:
        lo = params.get(f"min_{k}"); hi = params.get(f"max_{k}")
        if lo is not None and k in out.columns: out = out[out[k] >= float(lo)]
        if hi is not None and k in out.columns: out = out[out[k] <= float(hi)]
    return out.reset_index(drop=True)

# --------- Open‑Meteo (opcional): recálculo de dryness -> U(t) no detalhe ---------

TZ = "America/Sao_Paulo"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
if OM_AVAILABLE:
    import requests_cache
    cache_session = requests_cache.CachedSession('.cache', expire_after=1800)
    from retry_requests import retry
    retry_session = retry(cache_session, retries=3, backoff_factor=0.3)
    om_client = openmeteo_requests.Client(session=retry_session)

def compute_dryness_for_date(lat: float, lon: float, target_date: pd.Timestamp) -> Dict[str, float]:
    if not OM_AVAILABLE:
        return {"sm_norm": None, "et_scaled": None, "dryness": None}
    today = pd.Timestamp(date.today(), tz=timezone.utc).tz_convert(TZ).date()
    td = pd.to_datetime(target_date).date()
    if td < (today - timedelta(days=2)) or td > (today + timedelta(days=16)):
        return {"sm_norm": None, "et_scaled": None, "dryness": None}
    hourly_vars = ["evapotranspiration","soil_moisture_0_to_1cm"]
    params = {"latitude": lat, "longitude": lon, "timezone": TZ, "past_days": 2, "forecast_days": 16, "hourly": hourly_vars}
    resp = om_client.weather_api(FORECAST_URL, params=params)[0]
    h = resp.Hourly()
    times = pd.date_range(start=pd.to_datetime(h.Time(), unit="s", utc=True), end=pd.to_datetime(h.TimeEnd(), unit="s", utc=True),
                          freq=pd.Timedelta(seconds=h.Interval()), inclusive="left")
    df = pd.DataFrame({"time": times})
    for i in range(h.VariablesLength()):
        name = hourly_vars[i] if i < len(hourly_vars) else f"var_{i}"
        try: df[name] = h.Variables(i).ValuesAsNumpy()
        except Exception: df[name] = np.nan
    df["time_local"] = df["time"].dt.tz_convert(TZ); df["date"] = df["time_local"].dt.date
    et24 = float(df[df["date"] == td]["evapotranspiration"].sum()) if "evapotranspiration" in df else np.nan
    sm6  = float(df[df["date"] == td].tail(6)["soil_moisture_0_to_1cm"].mean()) if "soil_moisture_0_to_1cm" in df else np.nan
    sm_lo, sm_hi = 0.10, 0.45; et_lo, et_hi = 1.0, 6.0
    def clamp01(x): return float(max(0.0, min(1.0, x)))
    def scale(x, lo, hi): return clamp01((x - lo) / (hi - lo) if hi != lo else 0.0)
    sm_norm = scale(sm6, sm_lo, sm_hi) if sm6 == sm6 else None
    et_scaled = scale(et24, et_lo, et_hi) if et24 == et24 else None
    if sm_norm is None or et_scaled is None: return {"sm_norm": None, "et_scaled": None, "dryness": None}
    dryness = 0.5*(1.0 - sm_norm) + 0.5*et_scaled
    return {"sm_norm": sm_norm, "et_scaled": et_scaled, "dryness": clamp01(dryness)}

# ------------------------------ LLM / RAG helpers ------------------------------

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def _get_openai_client() -> OpenAI:
    if not OPENAI_SDK_OK:
        raise HTTPException(500, detail="Pacote 'openai' não instalado. pip install openai")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(500, detail="OPENAI_API_KEY ausente no ambiente.")
    return OpenAI(api_key=api_key)

def _load_population() -> dict:
    if not POP_CSV.exists(): return {}
    try:
        dfp = pd.read_csv(POP_CSV)
        if "bairro" not in dfp.columns:
            for c in dfp.columns:
                if c.lower().startswith("bair"):
                    dfp.rename(columns={c:"bairro"}, inplace=True); break
        if "population" not in dfp.columns:
            for c in dfp.columns:
                if "pop" in c.lower():
                    dfp.rename(columns={c:"population"}, inplace=True); break
        dfp["bairro"] = dfp["bairro"].astype(str)
        dfp["population"] = pd.to_numeric(dfp["population"], errors="coerce")
        dfp = dfp.dropna(subset=["bairro","population"])
        return dict(zip(dfp["bairro"], dfp["population"]))
    except Exception:
        return {}

def _playbook_text() -> str:
    return textwrap.dedent("""
    Diretrizes para análise (não inventar dados; usar apenas as métricas fornecidas):
    - Risco vermelho => ações imediatas (curto prazo). Risco amarelo => estruturais (médio/longo).
    - Se u_macro baixo: priorizar bombas/canais (operação, inspeção, desassoreamento, comportas).
    - Se u_micro baixo: limpeza de bocas de lobo, retirada de resíduos, correção de pontos de afogamento.
    - Se u_cobertura baixo: ampliar/adequar rede pluvial; checar subdimensionamentos.
    - Se u_permeabilidade baixo: infraestrutura verde (jardins de chuva, biovaletas, pav. permeável).
    - Perigo (H): P6/P1 altos => enxurradas; A72+SM altos => saturação; PP alto => confiança da previsão.
    - Se houver população estimada, mencionar exposição de forma sucinta e não alarmista.
    - Justificar sempre com o sub-índice mais fraco e/ou fator de perigo dominante.
    """)

def _build_bairro_context(rowU: pd.Series, H: float, risk_level: str, date_iso: str, pop_est: float | None) -> dict:
    U = float(rowU.get("U_t", rowU.get("U_static", 0)) or 0.0)
    risk = float(H * max(0.0, 1.0 - U))
    ctx = {
        "date": date_iso,
        "risk": {"score": round(risk, 3), "level": risk_level},
        "H_score": float(H),
        "U": U,
        "subindices": {
            "u_cobertura": float(rowU.get("u_cobertura", np.nan)) if "u_cobertura" in rowU else None,
            "u_micro": float(rowU.get("u_micro", np.nan)) if "u_micro" in rowU else None,
            "u_macro": float(rowU.get("u_macro", np.nan)) if "u_macro" in rowU else None,
            "u_permeabilidade": float(rowU.get("u_permeabilidade", np.nan)) if "u_permeabilidade" in rowU else None,
        },
        "infra_metrics": {
            "dens_pav_km_km2": float(rowU.get("dens_pav_km_km2", np.nan)) if "dens_pav_km_km2" in rowU else None,
            "dreno_km_km2": float(rowU.get("dreno_km_km2", np.nan)) if "dreno_km_km2" in rowU else None,
            "canal_km_km2": float(rowU.get("canal_km_km2", np.nan)) if "canal_km_km2" in rowU else None,
            "frac_verde": float(rowU.get("frac_verde", np.nan)) if "frac_verde" in rowU else None,
            "pumps_n": int(rowU.get("pumps_n", 0)) if "pumps_n" in rowU else None,
        },
        "population": {"estimate": float(pop_est) if pop_est is not None else None}
    }
    return ctx

def _insight_schema(max_actions: int, lang: str) -> dict:
    return {
        "type":"object",
        "properties":{
            "language":{"type":"string"},
            "summary":{"type":"string"},
            "alerts":{"type":"array","items":{
                "type":"object",
                "properties":{
                    "urgency":{"type":"string","enum":["immediate","short_term"]},
                    "message":{"type":"string"},
                    "why":{"type":"string"}
                },
                "required":["urgency","message"]
            }},
            "actions":{"type":"array","maxItems":max_actions,"items":{
                "type":"object",
                "properties":{
                    "title":{"type":"string"},
                    "urgency":{"type":"string","enum":["immediate","short_term"]},
                    "priority":{"type":"integer"},
                    "why":{"type":"string"},
                    "expected_impact":{"type":"string"}
                },
                "required":["title","urgency","priority"]
            }},
            "confidence":{"type":"number"}
        },
        "required":["language","summary","alerts","actions","confidence"]
    }

def _call_llm_insight(client: OpenAI, model: str, rag_text: str, context: dict, schema: dict, lang: str) -> dict:
    sys = f"Você é analista de defesa civil. Responda em {lang}. Devolva SOMENTE JSON válido que siga o schema."
    usr = {
        "role":"user",
        "content": textwrap.dedent(f"""
        ### RAG
        {rag_text}

        ### CONTEXTO
        {json.dumps(context, ensure_ascii=False)}

        ### SCHEMA
        {json.dumps(schema, ensure_ascii=False)}

        ### INSTRUÇÕES
        - Resumo objetivo do risco do bairro nesta data.
        - Alertas: 'immediate' (vermelho) ou 'short_term' (amarelo).
        - Até {schema['properties']['actions']['maxItems']} ações priorizadas.
        - Justifique com sub-índices mais fracos e fatores de perigo.
        - Se houver população, cite exposição de modo sucinto e não alarmista.
        """)
    }
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":sys}, usr],
        temperature=0.2,
        response_format={"type":"json_object"}
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {"language": lang, "summary":"", "alerts":[], "actions":[], "confidence":0.5}

# ----------------------------------- Endpoints --------------------------------

@app.get("/health")
def health():
    return {"status":"ok"}

@app.get("/v1/meta")
def meta():
    w = load_weights()
    dfH = try_load_hazard()
    return {
        "city": "canoas",
        "timezone": "America/Sao_Paulo",
        "hazard_date_min": dfH["date"].min().date().isoformat(),
        "hazard_date_max": dfH["date"].max().date().isoformat(),
        "thresholds": w["hazard_levels"],
        "weights": {"hazard": w["hazard_daily_weights"], "u": w["u_weights"]},
        "notes": "U==0 → no_data; ranking ignora no_data."
    }

# ---------------------- Risco em tabela (com filtros) -------------------------

@app.get("/v1/risk/by_bairro")
def risk_by_bairro(
    date_str: Optional[str] = Query(None, alias="date"),
    risk_level: Optional[str] = None,
    min_risk: Optional[float] = None, max_risk: Optional[float] = None,
    min_u_cobertura: Optional[float] = None, max_u_cobertura: Optional[float] = None,
    min_u_micro: Optional[float] = None, max_u_micro: Optional[float] = None,
    min_u_macro: Optional[float] = None, max_u_macro: Optional[float] = None,
    min_u_permeabilidade: Optional[float] = None, max_u_permeabilidade: Optional[float] = None,
    min_p6_pct: Optional[float] = None, max_p6_pct: Optional[float] = None,
    min_a72_pct: Optional[float] = None, max_a72_pct: Optional[float] = None,
    min_sm_norm: Optional[float] = None, max_sm_norm: Optional[float] = None,
    min_et_deficit: Optional[float] = None, max_et_deficit: Optional[float] = None,
    min_p1_pct: Optional[float] = None, max_p1_pct: Optional[float] = None,
    min_pp_unit: Optional[float] = None, max_pp_unit: Optional[float] = None,
    min_rd_norm: Optional[float] = None, max_rd_norm: Optional[float] = None,
):
    dfH = try_load_hazard()
    dfU, _ = try_load_u()
    thr = load_weights()["hazard_levels"]

    # Seleção de data
    df_sel = dfH if date_str is None else dfH[dfH["date"].dt.date == pd.to_datetime(date_str).date()]
    if df_sel.empty: df_sel = dfH[dfH["date"] == dfH["date"].max()]
    H = float(df_sel["H_score"].iloc[0]); d_sel = df_sel["date"].iloc[0]

    # Base por bairro
    df = dfU.copy()
    df["U"] = df.get("U_t", df.get("U_static", 0)).fillna(0)
    df["U_valid"] = df["U"] > 0
    df.loc[~df["U_valid"], ["Risk_score","Risk_level"]] = [np.nan,"no_data"]
    dfv = df[df["U_valid"]].copy()
    dfv["Fragilidade"] = 1 - dfv["U"]
    dfv["Risk_score"] = (H * dfv["Fragilidade"]).clip(0,1)
    dfv["Risk_level"] = dfv["Risk_score"].apply(lambda x: bucket_risk(x, thr))
    df.update(dfv)

    # Replica fatores de hazard (se existirem)
    rowH = dfH[dfH["date"]==d_sel].head(1)
    for k in ["p6_pct","a72_pct","sm_norm","et_deficit","p1_pct","pp_unit","rd_norm"]:
        if k in rowH.columns:
            df[k] = float(rowH.iloc[0][k]) if pd.notna(rowH.iloc[0][k]) else np.nan

    # Filtros
    params = {k: v for k, v in locals().items() if k.startswith("min_") or k.startswith("max_") or k=="risk_level"}
    df = apply_filters(df, params)

    df["date"] = d_sel.date().isoformat(); df["H_score"] = H
    cols = ["bairro","date","H_score","U","U_valid","Risk_score","Risk_level",
            "u_cobertura","u_micro","u_macro","u_permeabilidade",
            "p6_pct","a72_pct","sm_norm","et_deficit","p1_pct","pp_unit","rd_norm"]
    cols = [c for c in cols if c in df.columns]
    df = df.replace({np.nan: None, np.inf: None, -np.inf: None})  # <-- adiciona aqui
    return df[cols].to_dict(orient="records")


@app.get("/v1/risk/by_bairro/csv")
def risk_by_bairro_csv(date: Optional[str] = None):
    rows = risk_by_bairro(date_str=date)
    df = pd.DataFrame(rows)
    csv = df.to_csv(index=False)
    return PlainTextResponse(csv, media_type="text/csv; charset=utf-8")

@app.get("/v1/risk/by_bairro/top")
def risk_top(
    date: Optional[str] = Query(None, description="Data específica (YYYY-MM-DD)"),
    n: int = Query(5, ge=1, le=50)
):
    rows = risk_by_bairro(date_str=date)
    df = pd.DataFrame(rows)
    df = df[(df.get("U_valid", True)) & (df["Risk_score"].notna())]
    df = df.sort_values("Risk_score", ascending=False).head(n)
    df = df.replace({np.nan: None, np.inf: None, -np.inf: None})  # <-- adiciona aqui
    return df.to_dict(orient="records")

# --------------------------- Mapa GeoJSON por data ----------------------------

@app.get("/v1/geo/canoas/bairros_risk")
def geo_bairros_risk(
    date: Optional[str] = None,
    include: Optional[str] = Query("basic", description="basic|infra|hazard|all")
):
    dfH = try_load_hazard(); dfU, gdfU = try_load_u(); thr = load_weights()["hazard_levels"]
    df_sel = dfH if date is None else dfH[dfH["date"].dt.date == pd.to_datetime(date).date()]
    if df_sel.empty: df_sel = dfH[dfH["date"]==dfH["date"].max()]
    H = float(df_sel["H_score"].iloc[0]); d_sel = df_sel["date"].iloc[0]

    df = dfU.copy(); df["U"] = df.get("U_t", df.get("U_static", 0)).fillna(0)
    df["U_valid"] = df["U"] > 0
    df.loc[~df["U_valid"], ["Risk_score","Risk_level"]] = [np.nan,"no_data"]
    dfv = df[df["U_valid"]].copy()
    dfv["Fragilidade"] = 1 - dfv["U"]
    dfv["Risk_score"] = (H * dfv["Fragilidade"]).clip(0,1)
    dfv["Risk_level"] = dfv["Risk_score"].apply(lambda x: bucket_risk(x, thr))
    df.update(dfv)

    # Seleção de propriedades
    base_cols = ["bairro","U","U_valid","Risk_score","Risk_level"]
    infra_cols = ["u_cobertura","u_micro","u_macro","u_permeabilidade",
                  "dens_pav_km_km2","dreno_km_km2","canal_km_km2","frac_verde","pumps_n","area_km2"]
    hazard_cols = ["p6_pct","a72_pct","sm_norm","et_deficit","p1_pct","pp_unit","rd_norm"]
    # replica hazard cols se existirem
    rowH = dfH[dfH["date"]==d_sel].head(1)
    for k in hazard_cols:
        if k in rowH.columns:
            df[k] = float(rowH.iloc[0][k]) if pd.notna(rowH.iloc[0][k]) else np.nan
        else:
            if k in df.columns: df.drop(columns=[k], inplace=True)

    if include == "basic": props = base_cols
    elif include == "infra": props = list(dict.fromkeys(base_cols + infra_cols))
    elif include == "hazard": props = list(dict.fromkeys(base_cols + [c for c in hazard_cols if c in df.columns]))
    else: props = list(dict.fromkeys(base_cols + infra_cols + [c for c in hazard_cols if c in df.columns]))

    gdf = gdfU.merge(df[["bairro"] + [c for c in props if c != "bairro"]], on="bairro", how="left")
    gdf["date"] = d_sel.date().isoformat()
    # garante 'date' em string
    if "date" in gdf.columns: gdf["date"] = gdf["date"].astype(str)
    gj = json.loads(gdf[props + ["date","geometry"]].to_json())
    return gj

# --------------------------- Detalhe de um bairro -----------------------------

@app.get("/v1/bairros/detail")
def bairro_detail(
    bairro: str,
    date: Optional[str] = None,
    dynamic: int = Query(0, description="1 para tentar recalcular U(t) com Open-Meteo")
):
    dfH = try_load_hazard(); dfU, gdfU = try_load_u(); thr = load_weights()["hazard_levels"]
    df_sel = dfH if date is None else dfH[dfH["date"].dt.date == pd.to_datetime(date).date()]
    if df_sel.empty: df_sel = dfH[dfH["date"]==dfH["date"].max()]
    H = float(df_sel["H_score"].iloc[0]); d_sel = df_sel["date"].iloc[0]

    rowU = dfU[dfU["bairro"].astype(str)==str(bairro)].head(1)
    if rowU.empty: raise HTTPException(404, detail=f"Bairro '{bairro}' não encontrado.")
    U = float(rowU.get("U_t", rowU.get("U_static", 0)).iloc[0] or 0.0)
    # opcional: recálculo dinâmico via Open‑Meteo
    used_dynamic = False; dyn_info = {"sm_norm":None, "et_scaled":None, "dryness":None}
    if dynamic and OM_AVAILABLE and U > 0:
        rowG = gdfU[gdfU["bairro"].astype(str)==str(bairro)].head(1)
        if not rowG.empty:
            c = rowG.iloc[0].geometry.centroid
            res = compute_dryness_for_date(float(c.y), float(c.x), d_sel)
            if res["dryness"] is not None:
                U_static = float(rowU.get("U_static", rowU.get("U_t", 0)).iloc[0] or 0.0)
                U = float(np.clip(U_static + 0.10*(res["dryness"] - 0.5), 0, 1))
                used_dynamic = True; dyn_info = res

    if U == 0 or pd.isna(U):
        return {"bairro": bairro, "date": d_sel.date().isoformat(), "status": "no_data"}

    Frag = float(np.clip(1.0 - U, 0, 1)); Risk = float(np.clip(H * Frag, 0, 1))
    level = bucket_risk(Risk, thr)

    subs = {k: float(rowU.iloc[0][k]) for k in ["u_cobertura","u_micro","u_macro","u_permeabilidade"] if k in rowU.columns}
    metrics = {k: (float(rowU.iloc[0][k]) if k in rowU.columns and pd.notna(rowU.iloc[0][k]) else None)
               for k in ["dens_pav_km_km2","dreno_km_km2","canal_km_km2","frac_verde","pumps_n","area_km2"]}

    return {
        "city":"canoas", "bairro":bairro, "date": d_sel.date().isoformat(),
        "H_score": H, "U": U, "Fragilidade": Frag, "Risk_score": Risk, "Risk_level": level,
        "dynamic_used": used_dynamic, "dynamic_info": dyn_info,
        "u_subindices": subs, "infra_metrics": metrics
    }

# ------------------------------ Filtros (UI) ----------------------------------

@app.get("/v1/filters")
def filters_schema():
    return {
        "risk_level": {"type":"enum","values":["green","yellow","red","no_data"]},
        "score_range": {"type":"range","field":"Risk_score","min":0,"max":1},
        "hazard_factors": {
            "p6_pct":{"type":"range","min":0,"max":1},
            "a72_pct":{"type":"range","min":0,"max":1},
            "sm_norm":{"type":"range","min":0,"max":1},
            "et_deficit":{"type":"range","min":0,"max":1},
            "p1_pct":{"type":"range","min":0,"max":1},
            "pp_unit":{"type":"range","min":0,"max":1},
            "rd_norm":{"type":"range","min":0,"max":1}
        },
        "infra_subindices": {
            "u_cobertura":{"type":"range","min":0,"max":1},
            "u_micro":{"type":"range","min":0,"max":1},
            "u_macro":{"type":"range","min":0,"max":1},
            "u_permeabilidade":{"type":"range","min":0,"max":1}
        }
    }

# ------------------------------- INSIGHTS (LLM) -------------------------------

@app.get("/v1/insights/by_bairro")
def insights_by_bairro(
    bairro: str,
    date: Optional[str] = None,
    lang: str = Query("pt-BR"),
    max_actions: int = Query(5, ge=1, le=10),
    include_raw: int = Query(0, description="1 para incluir os dados usados (RAG)"),
):
    # Data base
    dfH = try_load_hazard(); dfU, _ = try_load_u(); thr = load_weights()["hazard_levels"]
    df_sel = dfH if date is None else dfH[dfH["date"].dt.date == pd.to_datetime(date).date()]
    if df_sel.empty: df_sel = dfH[dfH["date"]==dfH["date"].max()]
    H = float(df_sel["H_score"].iloc[0]); d_sel = df_sel["date"].iloc[0]

    rowU = dfU[dfU["bairro"].astype(str)==str(bairro)].head(1)
    if rowU.empty: raise HTTPException(404, detail=f"Bairro '{bairro}' não encontrado.")
    U = float(rowU.get("U_t", rowU.get("U_static", 0)).iloc[0] or 0.0)
    if U == 0 or pd.isna(U):
        return {"bairro": bairro, "date": d_sel.date().isoformat(), "status": "no_data"}

    Frag = float(np.clip(1.0 - U, 0, 1)); Risk = float(np.clip(H * Frag, 0, 1))
    level = bucket_risk(Risk, thr)

    pop_map = _load_population(); pop_est = pop_map.get(str(bairro))
    exposure = round(pop_est * Risk) if pop_est is not None else None

    client = _get_openai_client()
    rag_text = _playbook_text()
    schema = _insight_schema(max_actions=max_actions, lang=lang)
    context = _build_bairro_context(rowU.iloc[0], H, level, d_sel.date().isoformat(), pop_est)
    insight = _call_llm_insight(client, OPENAI_MODEL, rag_text, context, schema, lang)

    payload = {
        "city": "canoas",
        "bairro": bairro,
        "date": d_sel.date().isoformat(),
        "risk": {"score": round(Risk,3), "level": level},
        "population": {"estimate": pop_est, "exposure_estimate": exposure},
        "insight": insight
    }
    if include_raw: payload["inputs"] = context
    return JSONResponse(payload)

@app.get("/v1/insights/city_top")
def insights_city_top(
    date: Optional[str] = None,
    n: int = Query(5, ge=1, le=10),
    lang: str = Query("pt-BR")
):
    rows = risk_top(date=date, n=n)
    if not rows:
        return {"date": date, "items": [], "note": "sem dados para a data ou todos os bairros no_data"}

    client = _get_openai_client()
    rag = textwrap.dedent("""
    Objetivo: síntese para decisão operacional municipal.
    - Liste os bairros em ordem de risco com pontuação.
    - Recomende alocação de equipes e ações emergenciais rápidas.
    - Texto curto (<= 1200 caracteres).
    """)
    ctx = {"date": rows[0]["date"], "items": rows}
    schema = {"type":"object","properties":{
        "language":{"type":"string"},
        "summary":{"type":"string"},
        "prioritized_allocation":{"type":"array","items":{"type":"string"}}
    }, "required":["language","summary","prioritized_allocation"]}
    sys = f"Você é analista de operações municipais. Responda em {lang}. Devolva SOMENTE JSON."
    usr = {"role":"user","content": f"RAG:\n{rag}\n\nDATA:\n{json.dumps(ctx, ensure_ascii=False)}\n\nSCHEMA:\n{json.dumps(schema)}"}
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role":"system","content":sys}, usr],
        temperature=0.2,
        response_format={"type":"json_object"}
    )
    try: out = json.loads(resp.choices[0].message.content)
    except Exception: out = {"language": lang, "summary":"", "prioritized_allocation":[]}
    return JSONResponse({"date": ctx["date"], "n": len(rows), "insight": out, "items": rows})
