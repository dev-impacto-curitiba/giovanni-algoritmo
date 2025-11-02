# -*- coding: utf-8 -*-
# ================================
# Open-Meteo: Flood + Hourly Weather (RS) + H_score (Perigo Fluvial)
# ================================
import openmeteo_requests
import pandas as pd
import numpy as np
import requests_cache
from retry_requests import retry
from datetime import date

# -----------------
# Cliente com cache/retry
# -----------------
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.3)
om = openmeteo_requests.Client(session=retry_session)

FLOOD_URL   = "https://flood-api.open-meteo.com/v1/flood"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
TZ = "America/Sao_Paulo"

# -----------------
# Área/tempo
# -----------------
lat, lon = -30.03, -51.22  # Porto Alegre, RS
start_date = "2024-01-01"
end_date   = str(date.today())

# -----------------
# Variáveis HORÁRIAS (Open-Meteo)
# -----------------
hourly_vars = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation_probability",
    "precipitation",
    "rain",
    "showers",
    "weather_code",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "visibility",
    "evapotranspiration",
    "et0_fao_evapotranspiration",
    "vapor_pressure_deficit",
    "wind_speed_10m",
    "wind_speed_80m",
    "wind_speed_120m",
    "wind_speed_180m",
    "wind_direction_120m",
    "wind_direction_180m",
    "wind_gusts_10m",
    "temperature_80m",
    "temperature_120m",
    "temperature_180m",
    "soil_temperature_0cm",
    "soil_temperature_6cm",
    "soil_temperature_18cm",
    "soil_temperature_54cm",
    "soil_moisture_0_1cm",
    "soil_moisture_1_3cm",
    "soil_moisture_3_9cm",
    "soil_moisture_9_27cm",
    "soil_moisture_27_81cm",
]

def fetch_hourly_df(lat, lon, start_date, end_date, variables):
    """Baixa dados HORÁRIOS do Archive API e retorna DataFrame com 'time' (UTC) e 'time_local' tz-aware."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": TZ,
        "hourly": variables
    }
    responses = om.weather_api(ARCHIVE_URL, params=params)
    resp = responses[0]
    hourly = resp.Hourly()

    # time em UTC
    time_index = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )
    df = pd.DataFrame({"time": time_index})

    # colunas dinâmicas
    n = hourly.VariablesLength()
    for i in range(n):
        try:
            vals = hourly.Variables(i).ValuesAsNumpy()
            name = variables[i] if i < len(variables) else f"var_{i}"
            df[name] = vals
        except Exception:
            pass

    # timezone local (tz-aware) e chave diária
    df["time_local"] = df["time"].dt.tz_convert(TZ)
    df["date"] = df["time_local"].dt.date
    return df

# -----------------
# Variáveis DIÁRIAS (Flood API)
# -----------------
flood_daily_vars = [
    "river_discharge",
    "river_discharge_mean",
    "river_discharge_median",
    "river_discharge_min",
    "river_discharge_max",
    "river_discharge_p25",
    "river_discharge_p75"
]

def fetch_flood_daily_df(lat, lon, start_date, end_date, variables):
    """Baixa dados DIÁRIOS de descarga (Flood API) e retorna DataFrame diário."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": TZ,
        "daily": variables
    }
    responses = om.weather_api(FLOOD_URL, params=params)
    resp = responses[0]
    daily = resp.Daily()

    date_index = pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left"
    ).tz_convert(TZ).date

    out = pd.DataFrame({"date": date_index})
    for i in range(daily.VariablesLength()):
        col = variables[i] if i < len(variables) else f"flood_var_{i}"
        try:
            out[col] = daily.Variables(i).ValuesAsNumpy()
        except Exception:
            pass

    out["latitude"] = resp.Latitude()
    out["longitude"] = resp.Longitude()
    return out

# -----------------
# Agregação: horário -> diário (para casar com Flood)
# -----------------
def agg_hourly_to_daily(weather_hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa por 'date' com regras sensatas: soma de precip/ET; máx de rajada; médias do restante."""
    df = weather_hourly_df.copy()

    sum_cols = [c for c in df.columns if c in {
        "precipitation", "rain", "showers",
        "evapotranspiration", "et0_fao_evapotranspiration"
    }]
    max_cols = [c for c in df.columns if c in {"wind_gusts_10m"}]
    mean_cols = [c for c in df.columns
                 if c not in (["time","time_local","date"] + sum_cols + max_cols)
                 and c != "weather_code"]

    agg = {}
    for c in sum_cols:  agg[c] = "sum"
    for c in max_cols:  agg[c] = "max"
    for c in mean_cols: agg[c] = "mean"

    if "weather_code" in df.columns:
        def mode_or_nan(x):
            m = x.mode()
            return m.iloc[0] if not m.empty else np.nan
        agg["weather_code"] = mode_or_nan

    daily = df.groupby("date", as_index=False).agg(agg)
    return daily

# -----------------
# === H_score: features, normalização e ponderação ===
# -----------------
def make_h_features(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """Cria P1/P6/A72/PP/SM/ET por dia a partir do horário (corrigido: bounds tz-aware)."""
    dfh = df_hourly.sort_values("time").copy()
    dfh["date"] = dfh["time_local"].dt.date

    precip = dfh.get("precipitation", pd.Series(0.0, index=dfh.index)).fillna(0.0)
    roll6 = precip.rolling(window=6, min_periods=1).sum()

    dates = np.array(sorted(dfh["date"].unique()))
    rows = []
    for d in dates:
        # recorte do dia (usando coluna tz-aware time_local)
        start_local = pd.Timestamp(d).tz_localize(TZ)
        end_local   = start_local + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        mask_day = (dfh["time_local"] >= start_local) & (dfh["time_local"] <= end_local)

        day_precip = precip[mask_day]
        day_pp = dfh.loc[mask_day, "precipitation_probability"] if "precipitation_probability" in dfh.columns else pd.Series([], dtype=float)

        # P1/P6
        p1_mm = float(day_precip.max()) if not day_precip.empty else 0.0
        p6_mm = float(roll6[mask_day].max()) if mask_day.any() else 0.0

        # PP (0..1)
        pp_max = float(day_pp.max()/100.0) if not day_pp.empty else 0.0

        # A72: janela [end_local-72h, end_local] — comparar em UTC com dfh["time"] (UTC)
        end_dt_utc   = end_local.tz_convert("UTC")
        start_dt_utc = end_dt_utc - pd.Timedelta(hours=72)
        mask72 = (dfh["time"] > start_dt_utc) & (dfh["time"] <= end_dt_utc)
        a72_mm = float(precip[mask72].sum()) if mask72.any() else 0.0

        # SM: média da camada mais rasa disponível no dia (usar time_local bounds tz-aware)
        sm_cols = [c for c in ["soil_moisture_0_1cm","soil_moisture_1_3cm","soil_moisture_3_9cm"] if c in dfh.columns]
        if sm_cols:
            sm_vals = dfh.loc[mask_day, sm_cols[0]].astype(float)
            sm_mean = float(sm_vals.mean()) if not sm_vals.empty else np.nan
        else:
            sm_mean = np.nan

        # ET diária
        et24 = float(dfh.loc[mask_day, "evapotranspiration"].sum()) if "evapotranspiration" in dfh.columns else np.nan

        rows.append({
            "date": d,
            "p1_mm": p1_mm,
            "p6_mm": p6_mm,
            "a72_mm": a72_mm,
            "pp_max": pp_max,
            "sm_mean": (sm_mean if sm_mean==sm_mean else np.nan),
            "et24_mm": (et24 if et24==et24 else np.nan),
        })
    return pd.DataFrame(rows)

def percentile_norm(series: pd.Series, values: pd.Series) -> pd.Series:
    arr = np.asarray(series.dropna().values, dtype=float)
    if arr.size == 0:
        return pd.Series([np.nan]*len(values), index=values.index)
    return values.apply(lambda x: float(np.mean(arr <= float(x))) if pd.notna(x) else np.nan)

def scale_deficit(series: pd.Series, lo: float, hi: float) -> pd.Series:
    s = (series - lo) / (hi - lo)
    s = s.clip(lower=0, upper=1)
    return 1.0 - s

def daily_weights(ribeirinho: bool=False) -> dict:
    # Pesos base (equilíbrio 6–24h). Re-normalizamos se faltar termo / RD.
    w = {"p6":0.25, "a72":0.25, "sm":0.15, "etd":0.10, "p1":0.10, "pp":0.05}
    if ribeirinho:
        w["rd"] = 0.10
        s = sum(w.values())
        w = {k: v/s for k,v in w.items()}
    return w

def compute_h_score(feats: pd.DataFrame, flood_daily: pd.DataFrame) -> pd.DataFrame:
    # baselines
    p1_base  = feats["p1_mm"]
    p6_base  = feats["p6_mm"]
    a72_base = feats["a72_mm"]
    sm_base  = feats["sm_mean"].dropna() if "sm_mean" in feats.columns else pd.Series(dtype=float)
    et_base  = feats["et24_mm"].dropna() if "et24_mm" in feats.columns else pd.Series(dtype=float)
    rd_base  = flood_daily["river_discharge"].dropna() if "river_discharge" in flood_daily.columns else pd.Series(dtype=float)

    feats["p1_pct"]  = percentile_norm(p1_base, feats["p1_mm"]).clip(0,1)
    feats["p6_pct"]  = percentile_norm(p6_base, feats["p6_mm"]).clip(0,1)
    feats["a72_pct"] = percentile_norm(a72_base, feats["a72_mm"]).clip(0,1)
    feats["pp_unit"] = feats["pp_max"].clip(lower=0, upper=1)

    feats["sm_norm"] = percentile_norm(sm_base, feats["sm_mean"]).clip(0,1) if not sm_base.empty else np.nan
    feats["et_deficit"] = scale_deficit(feats["et24_mm"], lo=1.0, hi=6.0) if not et_base.empty else np.nan

    # RD (opcional)
    if not rd_base.empty and "river_discharge" in flood_daily.columns:
        feats = feats.merge(flood_daily[["date","river_discharge"]], on="date", how="left")
        feats["rd_norm"] = percentile_norm(rd_base, feats["river_discharge"]).clip(0,1)
        ribeirinho = True
    else:
        feats["rd_norm"] = np.nan
        ribeirinho = False

    W = daily_weights(ribeirinho=ribeirinho)

    def row_score(r):
        weights = W.copy()
        terms = {"p6": r["p6_pct"], "a72": r["a72_pct"], "sm": r["sm_norm"],
                 "etd": r["et_deficit"], "p1": r["p1_pct"], "pp": r["pp_unit"],
                 "rd": r.get("rd_norm", np.nan)}
        # re-normaliza com base no que existe
        avail = {k: pd.notna(terms[k]) for k in weights}
        s = sum(weights[k] for k in weights if avail[k])
        weights = {k: (weights[k]/s if avail[k] and s>0 else 0.0) for k in weights}
        total = 0.0
        for k,w in weights.items():
            v = terms.get(k, 0.0)
            if pd.notna(v):
                total += w*float(v)
        return max(0.0, min(1.0, total))

    feats["H_score"] = feats.apply(row_score, axis=1)
    return feats

# -----------------
# Execução
# -----------------
if __name__ == "__main__":
    # 1) Meteorologia/solo horário -> diário
    wx_hourly = fetch_hourly_df(lat, lon, start_date, end_date, hourly_vars)
    wx_daily  = agg_hourly_to_daily(wx_hourly)
    wx_daily.to_csv("rs_weather_daily.csv", index=False)

    # 2) Hidrologia diária (Flood)
    flood_daily = fetch_flood_daily_df(lat, lon, start_date, end_date, flood_daily_vars)
    flood_daily.to_csv("rs_flood_daily.csv", index=False)

    # 3) H_score (features + normalização + ponderação)
    feats = make_h_features(wx_hourly)
    feats = compute_h_score(feats, flood_daily)
    feats.to_csv("rs_hazard_daily.csv", index=False)

    # 4) Merge completo (hidrologia + H_score)
    merged = flood_daily.merge(feats, on="date", how="left")
    merged.to_csv("rs_flood_weather_hazard.csv", index=False)

    print(f"Salvos: rs_weather_daily.csv ({len(wx_daily)} linhas), "
          f"rs_flood_daily.csv ({len(flood_daily)} linhas), "
          f"rs_hazard_daily.csv ({len(feats)} linhas), "
          f"rs_flood_weather_hazard.csv ({len(merged)} linhas))")
