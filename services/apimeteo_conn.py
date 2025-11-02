# -*- coding: utf-8 -*-
# ==========================================
# Open-Meteo: Previsão de Perigo Fluvial (H_score) até 16 dias
# ==========================================
import openmeteo_requests
import pandas as pd
import numpy as np
import requests_cache
from retry_requests import retry
from datetime import date

# -----------------------
# Configuração da API
# -----------------------
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.3)
om = openmeteo_requests.Client(session=retry_session)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FLOOD_URL    = "https://flood-api.open-meteo.com/v1/flood"
TZ = "America/Sao_Paulo"

# -----------------------
# Local e horizonte
# -----------------------
lat, lon = -30.03, -51.22  # Porto Alegre, RS
forecast_days = 16
today = str(date.today())

# -----------------------
# Variáveis meteorológicas
# -----------------------
hourly_vars = [
    "precipitation", "precipitation_probability",
    "soil_moisture_0_to_1cm", "soil_moisture_1_to_3cm",
    "evapotranspiration"
]

# -----------------------
# Variáveis do Flood API
# -----------------------
flood_daily_vars = [
    "river_discharge",
    "river_discharge_mean",
    "river_discharge_max",
    "river_discharge_min"
]

# -----------------------
# 1) Previsão horária (meteo)
# -----------------------
def fetch_forecast_hourly(lat, lon, forecast_days, variables):
    params = {
        "latitude": lat,
        "longitude": lon,
        "forecast_days": forecast_days,
        "timezone": TZ,
        "hourly": variables
    }
    responses = om.weather_api(FORECAST_URL, params=params)
    resp = responses[0]
    hourly = resp.Hourly()

    # Cria DataFrame base
    time_index = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )
    df = pd.DataFrame({"time": time_index})

    n = hourly.VariablesLength()
    for i in range(n):
        try:
            vals = hourly.Variables(i).ValuesAsNumpy()
            name = variables[i] if i < len(variables) else f"var_{i}"
            df[name] = vals
        except Exception:
            pass

    # ⚙️ Fallback automático para umidade do solo
    sm_var = "soil_moisture_0_to_1cm"
    if sm_var not in df.columns:
        print("⚠️ Variável soil_moisture_0_to_1cm ausente — criando com valor neutro (0.5)")
        df[sm_var] = 0.5
    else:
        if df[sm_var].isna().all() or df[sm_var].sum() == 0:
            print("⚠️ Nenhum dado de umidade do solo — usando fallback neutro (0.5)")
            df[sm_var] = 0.5
        else:
            # substitui eventuais NaN individuais
            df[sm_var] = df[sm_var].fillna(0.5)

    # 🧮 Garante sm_norm = 0.5 se nada vier
    if "soil_moisture_0_to_1cm" not in df.columns:
        df["sm_norm"] = 0.5
    else:
        sm_lo, sm_hi = 0.10, 0.45
        def scale_linear(x, lo, hi):
            return np.clip((x - lo) / (hi - lo), 0, 1)
        df["sm_norm"] = scale_linear(df["soil_moisture_0_to_1cm"], sm_lo, sm_hi)
        df["sm_norm"] = df["sm_norm"].fillna(0.5)

    # Normaliza timezone e retorna
    df["time_local"] = df["time"].dt.tz_convert(TZ)
    df["date"] = df["time_local"].dt.date
    return df


# -----------------------
# 2) Flood API diária (forecast)
# -----------------------
def fetch_forecast_flood(lat, lon, forecast_days, variables):
    params = {
        "latitude": lat,
        "longitude": lon,
        "forecast_days": forecast_days,
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


# -----------------------
# 3) Agregar hora → dia (features diárias)
# -----------------------
def daily_features_from_hourly(dfh: pd.DataFrame) -> pd.DataFrame:
    dfh = dfh.sort_values("time")
    dfh["date"] = dfh["time_local"].dt.date
    precip = dfh["precipitation"].fillna(0.0)
    roll6 = precip.rolling(window=6, min_periods=1).sum()

    dates = sorted(dfh["date"].unique())
    rows = []
    for d in dates:
        start_local = pd.Timestamp(d).tz_localize(TZ)
        end_local   = start_local + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        mask_day = (dfh["time_local"] >= start_local) & (dfh["time_local"] <= end_local)

        day_precip = precip[mask_day]
        p1_mm = float(day_precip.max()) if not day_precip.empty else 0.0
        p6_mm = float(roll6[mask_day].max()) if mask_day.any() else 0.0

        pp_max = float(dfh.loc[mask_day, "precipitation_probability"].max()/100.0) if "precipitation_probability" in dfh.columns else 0.0
        sm_mean = float(dfh.loc[mask_day, "soil_moisture_0_to_1cm"].mean()) if "soil_moisture_0_to_1cm" in dfh.columns else 0.5
        et24 = float(dfh.loc[mask_day, "evapotranspiration"].sum()) if "evapotranspiration" in dfh.columns else np.nan

        rows.append({"date": d, "p1_mm": p1_mm, "p6_mm": p6_mm, "pp_max": pp_max, "sm_mean": sm_mean, "et24_mm": et24})
    return pd.DataFrame(rows)


# -----------------------
# 4) Normalização e H_score
# -----------------------
def percentile_norm(series: pd.Series, values: pd.Series) -> pd.Series:
    arr = np.asarray(series.dropna().values, dtype=float)
    if arr.size == 0:
        return pd.Series([np.nan]*len(values), index=values.index)
    return values.apply(lambda x: float(np.mean(arr <= float(x))) if pd.notna(x) else np.nan)

def scale_deficit(series: pd.Series, lo: float, hi: float) -> pd.Series:
    s = (series - lo) / (hi - lo)
    s = s.clip(lower=0, upper=1)
    return 1.0 - s

def daily_weights(ribeirinho=False):
    w = {"p6":0.25,"a72":0.25,"sm":0.15,"etd":0.10,"p1":0.10,"pp":0.05}
    if ribeirinho:
        w["rd"] = 0.10
        s = sum(w.values())
        w = {k:v/s for k,v in w.items()}
    return w

def compute_h_score(forecast_df: pd.DataFrame, flood_df: pd.DataFrame):
    feats = forecast_df.copy()
    p1_base = feats["p1_mm"]; p6_base = feats["p6_mm"]
    sm_base = feats["sm_mean"].fillna(0.5)
    et_base = feats["et24_mm"].dropna() if "et24_mm" in feats else pd.Series(dtype=float)
    rd_base = flood_df["river_discharge"].dropna() if "river_discharge" in flood_df else pd.Series(dtype=float)

    feats["p1_pct"] = percentile_norm(p1_base, feats["p1_mm"]).clip(0,1)
    feats["p6_pct"] = percentile_norm(p6_base, feats["p6_mm"]).clip(0,1)
    feats["pp_unit"] = feats["pp_max"].clip(0,1)
    feats["sm_norm"] = 0.5  # garante sm_norm neutro
    feats["et_deficit"] = scale_deficit(feats["et24_mm"],1.0,6.0) if not et_base.empty else np.nan

    if not rd_base.empty:
        feats = feats.merge(flood_df[["date","river_discharge"]],on="date",how="left")
        feats["rd_norm"] = percentile_norm(rd_base,feats["river_discharge"]).clip(0,1)
        ribeirinho=True
    else:
        feats["rd_norm"]=np.nan
        ribeirinho=False

    W=daily_weights(ribeirinho)
    def row_score(r):
        weights=W.copy()
        terms={"p6":r["p6_pct"],"a72":r["p6_pct"],"sm":r["sm_norm"],"etd":r["et_deficit"],
               "p1":r["p1_pct"],"pp":r["pp_unit"],"rd":r.get("rd_norm",np.nan)}
        avail={k:pd.notna(terms[k]) for k in weights}
        s=sum(weights[k] for k in weights if avail[k])
        weights={k:(weights[k]/s if avail[k] and s>0 else 0.0) for k in weights}
        total=sum(weights[k]*float(terms[k]) for k in weights if pd.notna(terms[k]))
        return max(0.0,min(1.0,total))
    feats["H_score"]=feats.apply(row_score,axis=1)
    return feats


# -----------------------
# Execução principal
# -----------------------
if __name__=="__main__":
    print("🌦️ Baixando previsão de 16 dias da Open-Meteo...")
    wx_hourly = fetch_forecast_hourly(lat, lon, forecast_days, hourly_vars)
    flood_daily = fetch_forecast_flood(lat, lon, forecast_days, flood_daily_vars)

    print("📅 Gerando features diárias e H_score...")
    feats = daily_features_from_hourly(wx_hourly)
    feats = compute_h_score(feats, flood_daily)

    wx_hourly.to_csv("./data/hazard/weather_forecast_hourly.csv",index=False)
    feats.to_csv("./data/hazard/hazard_forecast.csv",index=False)
    flood_daily.to_csv("./data/hazard/flood_forecast.csv",index=False)

    merged = flood_daily.merge(feats,on="date",how="left")
    merged.to_csv("./data/hazard/flood_weather_hazard_forecast.csv",index=False)

    print("✅ Arquivos salvos:")
    print(" - weather_forecast_hourly.csv")
    print(" - flood_forecast.csv")
    print(" - hazard_forecast.csv")
    print(" - flood_weather_hazard_forecast.csv")
    print("\nPrévia:")
    print(merged[["date","river_discharge","p6_mm","pp_max","H_score"]].head())
