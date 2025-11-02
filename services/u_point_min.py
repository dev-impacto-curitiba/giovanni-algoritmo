# -*- coding: utf-8 -*-
"""
Gera U por bairro (Canoas/RS) com OSM + Open‑Meteo e exporta GeoJSON + CSV.

Passos:
1) Baixa/usa cache do GeoJSON oficial dos bairros (GeoCanoas, ArcGIS REST).
2) P/ cada bairro, busca OSM (Overpass) no bbox, intersecta com o polígono:
   - vias pavimentadas (km)
   - drenos/valas (km)
   - canais (km)
   - áreas verdes (km²)
   - bombas (contagem)
3) Normaliza e calcula sub-índices + U_static.
4) Open-Meteo no centróide p/ dryness (sm + ET) -> U(t), Fragilidade(t)
5) Exporta: data/u/canoas_bairros_u.geojson e data/u/canoas_bairros_u.csv
"""

import json
from pathlib import Path
from typing import Dict, Any, List

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import requests_cache
from retry_requests import retry
from shapely.geometry import LineString, Polygon, Point, shape
from shapely.ops import unary_union
from pyproj import Transformer
import openmeteo_requests
from datetime import date

# ------------------ Config ------------------

DATA_DIR = Path("data/u")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# GeoCanoas endpoints (ambos suportam f=geojson)
GEO_URLS = [
    "https://geo.canoas.rs.gov.br/server/rest/services/Hosted/Populacao_e_domicilios_por_bairros/FeatureServer/0/query?where=1=1&outFields=*&outSR=4326&f=geojson",
    "https://geo.canoas.rs.gov.br/server/rest/services/Covid_Canoas/FeatureServer/2/query?where=1=1&outFields=*&outSR=4326&f=geojson",
]

# Overpass mirrors
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter"
]

# Open‑Meteo client with cache/retry
cache_session = requests_cache.CachedSession('.cache', expire_after=1800)
retry_session = retry(cache_session, retries=3, backoff_factor=0.3)
om = openmeteo_requests.Client(session=retry_session)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TZ = "America/Sao_Paulo"

# Pesos U_min (re-normaliza se algo faltar)
WEIGHTS = {"perm": 0.40, "macro": 0.25, "cob": 0.20, "micro": 0.15}

# Âncoras de normalização (edite conforme calibração local)
ANCHORS = {
    "dens_pav_km_km2": (4.0, 18.0),   # vias pavimentadas por km²
    "dreno_km_km2":    (0.05, 0.50),  # dreno/vala por km²
    "canal_km_km2":    (0.10, 1.00),  # canal por km²
    "frac_verde":      (0.05, 0.30),  # fração verde no bairro
    "sm_clamp":        (0.10, 0.45),  # umidade solo (m3/m3)
    "et_day":          (1.0, 6.0)     # ET diária (mm)
}
DELTA_DRYNESS = 0.10  # ganho dinâmico

# ------------------ Utils ------------------

def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def scale_linear(x: float, lo: float, hi: float, invert: bool = False) -> float:
    if hi == lo:
        return 0.0
    s = (x - lo) / (hi - lo)
    s = clamp01(s)
    return 1.0 - s if invert else s

def projectors(lat: float, lon: float):
    """
    Usa UTM 22S (EPSG:31982) para Canoas/RS; fallback: WebMercator.
    """
    try:
        to_xy = Transformer.from_crs("EPSG:4326", "EPSG:31982", always_xy=True)
        to_ll = Transformer.from_crs("EPSG:31982", "EPSG:4326", always_xy=True)
        return to_xy, to_ll
    except Exception:
        to_xy = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        to_ll = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        return to_xy, to_ll

def ensure_bairros_geojson(out_path: Path) -> None:
    if out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for url in GEO_URLS:
        try:
            r = requests.get(url, timeout=60)
            if r.ok and "json" in r.headers.get("content-type","").lower():
                out_path.write_bytes(r.content)
                return
        except Exception:
            continue
    raise RuntimeError("Não foi possível baixar o GeoJSON de bairros do GeoCanoas.")

def overpass(query: str) -> Dict[str, Any]:
    for u in OVERPASS_URLS:
        try:
            r = requests.post(u, data={"data": query}, timeout=120)
            if r.ok:
                return r.json()
        except Exception:
            continue
    raise RuntimeError("Overpass sem resposta (tente novamente).")

def lines_length_km(elems: List[Dict[str,Any]], poly_xy, to_xy) -> float:
    total = 0.0
    for e in elems:
        if "geometry" not in e:
            continue
        coords = [(p["lon"], p["lat"]) for p in e["geometry"]]
        if len(coords) < 2:
            continue
        xys = [to_xy.transform(x, y) for x, y in coords]
        line = LineString(xys)
        inter = line.intersection(poly_xy)
        if not inter.is_empty:
            total += inter.length
    return total / 1000.0

def polygons_area_km2(elems: List[Dict[str,Any]], poly_xy, to_xy) -> float:
    total = 0.0
    for e in elems:
        if "geometry" not in e:
            continue
        coords = [(p["lon"], p["lat"]) for p in e["geometry"]]
        if len(coords) < 3:
            continue
        xys = [to_xy.transform(x, y) for x, y in coords]
        geom = Polygon(xys)
        if not geom.is_valid:
            geom = geom.buffer(0)
        inter = geom.intersection(poly_xy)
        if not inter.is_empty:
            total += inter.area
    return total / 1e6

def point_in_poly_count(nodes: List[Dict[str,Any]], poly_xy, to_xy) -> int:
    count = 0
    for n in nodes:
        if "lon" in n and "lat" in n:
            x,y = to_xy.transform(n["lon"], n["lat"])
            if Point(x,y).within(poly_xy):
                count += 1
    return count

def fetch_osm_metrics_for_polygon(poly_wgs: Polygon) -> Dict[str,float]:
    # bbox em WGS84
    minx, miny, maxx, maxy = poly_wgs.bounds  # lon/lat
    # consulta Overpass no bbox
    q = f"""
    [out:json][timeout:60];
    (
      way({miny},{minx},{maxy},{maxx})["highway"]["surface"~"asphalt|paved|concrete"];
      way({miny},{minx},{maxy},{maxx})["waterway"~"drain|ditch"];
      way({miny},{minx},{maxy},{maxx})["waterway"="canal"];
      way({miny},{minx},{maxy},{maxx})["landuse"~"grass|forest|meadow|recreation_ground|park"];
      way({miny},{minx},{maxy},{maxx})["natural"~"wood|scrub|grassland|heath|wetland"];
      way({miny},{minx},{maxy},{maxx})["leisure"="park"];
      node({miny},{minx},{maxy},{maxx})["man_made"="pumping_station"];
    );
    out tags geom;
    """
    data = overpass(q)
    elements = data.get("elements", [])

    paved = [e for e in elements if e.get("type") == "way"  and e.get("tags",{}).get("highway") and e.get("tags",{}).get("surface")]
    drain = [e for e in elements if e.get("type") == "way"  and e.get("tags",{}).get("waterway") in ("drain","ditch")]
    canal = [e for e in elements if e.get("type") == "way"  and e.get("tags",{}).get("waterway") == "canal"]
    greens= [e for e in elements if e.get("type") == "way"  and (
                e.get("tags",{}).get("landuse") in ("grass","forest","meadow","recreation_ground","park")
             or e.get("tags",{}).get("natural") in ("wood","scrub","grassland","heath","wetland")
             or e.get("tags",{}).get("leisure") == "park")]
    pumps = [e for e in elements if e.get("type") == "node" and e.get("tags",{}).get("man_made") == "pumping_station"]

    # projeta para CRS métrico
    # usa centróide para escolher projeção local
    c = poly_wgs.centroid
    to_xy, _ = projectors(c.y, c.x)
    poly_xy = Polygon([to_xy.transform(x,y) for x,y in np.array(poly_wgs.exterior.coords)])

    area_km2 = poly_xy.area / 1e6
    paved_km  = lines_length_km(paved, poly_xy, to_xy)
    drain_km  = lines_length_km(drain, poly_xy, to_xy)
    canal_km  = lines_length_km(canal, poly_xy, to_xy)
    green_km2 = polygons_area_km2(greens, poly_xy, to_xy)
    pumps_n   = point_in_poly_count(pumps, poly_xy, to_xy)

    return {
        "area_km2": area_km2,
        "paved_km": paved_km,
        "drain_km": drain_km,
        "canal_km": canal_km,
        "green_km2": green_km2,
        "pumps_n": pumps_n
    }

def fetch_dryness(lat: float, lon: float) -> Dict[str,Any]:
    hourly_vars = ["evapotranspiration","soil_moisture_0_to_1cm"]
    params = {
        "latitude": lat, "longitude": lon, "timezone": TZ,
        "forecast_days": 1, "past_days": 2, "hourly": hourly_vars
    }
    resp = om.weather_api(FORECAST_URL, params=params)[0]
    h = resp.Hourly()
    times = pd.date_range(
        start=pd.to_datetime(h.Time(), unit="s", utc=True),
        end=pd.to_datetime(h.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=h.Interval()),
        inclusive="left"
    )
    df = pd.DataFrame({"time": times})
    for i in range(h.VariablesLength()):
        name = hourly_vars[i] if i < len(hourly_vars) else f"var_{i}"
        try:
            df[name] = h.Variables(i).ValuesAsNumpy()
        except Exception:
            df[name] = np.nan
    df["time_local"] = df["time"].dt.tz_convert(TZ)
    df["date"] = df["time_local"].dt.date

    # ET do último dia completo
    last_day = df["date"].iloc[-1]
    et24 = float(df[df["date"] == last_day]["evapotranspiration"].sum()) if "evapotranspiration" in df else np.nan
    # SM média das últimas 6h
    sm6  = float(df.tail(6)["soil_moisture_0_to_1cm"].mean()) if "soil_moisture_0_to_1cm" in df else np.nan

    sm_lo, sm_hi = ANCHORS["sm_clamp"]
    et_lo, et_hi = ANCHORS["et_day"]
    sm_norm = scale_linear(sm6, sm_lo, sm_hi) if sm6 == sm6 else 0.5
    et_scaled = scale_linear(et24, et_lo, et_hi) if et24 == et24 else 0.5
    dryness = 0.5*(1.0 - sm_norm) + 0.5*et_scaled

    return {
        "calc_date": str(last_day),
        "et24_mm": None if et24 != et24 else et24,
        "sm6_m3m3": None if sm6 != sm6 else sm6,
        "sm_norm": clamp01(sm_norm),
        "et_scaled": clamp01(et_scaled),
        "dryness": clamp01(dryness)
    }

def compute_u_from_metrics(metrics: Dict[str,float], centroid_lat: float, centroid_lon: float) -> Dict[str,Any]:
    area = max(1e-6, metrics["area_km2"])
    dens_pav = metrics["paved_km"] / area
    dreno_km2 = metrics["drain_km"] / area
    canal_km2 = metrics["canal_km"] / area
    frac_verde = min(1.0, metrics["green_km2"] / area)
    has_pump = 1.0 if metrics["pumps_n"] > 0 else 0.0

    u_cob  = scale_linear(dens_pav, *ANCHORS["dens_pav_km_km2"])
    u_micro= scale_linear(dreno_km2, *ANCHORS["dreno_km_km2"])
    u_macro= clamp01(0.5*has_pump + 0.5*scale_linear(canal_km2, *ANCHORS["canal_km_km2"]))
    u_perm = scale_linear(frac_verde, *ANCHORS["frac_verde"])

    weights = WEIGHTS.copy()
    # (todos disponíveis neste fluxo) re-normalização defensiva
    s = sum(weights.values())
    weights = {k: v/s for k,v in weights.items()}

    U_static = clamp01(weights["perm"]*u_perm + weights["macro"]*u_macro +
                       weights["cob"]*u_cob + weights["micro"]*u_micro)

    dyn = fetch_dryness(centroid_lat, centroid_lon)
    dryness = dyn["dryness"]
    U_t = clamp01(U_static + DELTA_DRYNESS*(dryness - 0.5))
    frag_t = clamp01(1.0 - U_t)

    return {
        "densities": {
            "dens_pav_km_km2": round(dens_pav,3),
            "dreno_km_km2": round(dreno_km2,3),
            "canal_km_km2": round(canal_km2,3),
            "frac_verde": round(frac_verde,3)
        },
        "subindices": {
            "u_cobertura": round(u_cob,3),
            "u_micro": round(u_micro,3),
            "u_macro": round(u_macro,3),
            "u_permeabilidade": round(u_perm,3)
        },
        "weights": weights,
        "U_static": round(U_static,3),
        "dynamic": dyn,
        "U_t": round(U_t,3),
        "Fragilidade_t": round(frag_t,3)
    }

# ------------------ Main flow ------------------

def main():
    # 1) GeoJSON de bairros (cache local)
    gj_path = DATA_DIR / "canoas_bairros.geojson"
    ensure_bairros_geojson(gj_path)

    gdf = gpd.read_file(gj_path)
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    else:
        gdf = gdf.to_crs(epsg=4326)

    # Campos de identificação usuais
    name_field = None
    for cand in ["bairro","name","NOME","BAIRRO","Bairro"]:
        if cand in gdf.columns:
            name_field = cand
            break
    if name_field is None:
        name_field = "OBJECTID"  # fallback

    rows = []
    geoms = []
    for idx, row in gdf.iterrows():
        geom_wgs = row.geometry
        if geom_wgs is None or geom_wgs.is_empty:
            continue
        # força polígono simples (trata MultiPolygon)
        if geom_wgs.geom_type == "MultiPolygon":
            geom_wgs = unary_union([poly for poly in geom_wgs.geoms if poly.area > 0])
        if geom_wgs.is_empty:
            continue

        # 2) Métricas OSM por polígono
        metrics = fetch_osm_metrics_for_polygon(geom_wgs)

        # 3) Centróide para dinâmica Open‑Meteo
        c = geom_wgs.centroid
        ures = compute_u_from_metrics(metrics, centroid_lat=c.y, centroid_lon=c.x)

        props = {
            "bairro": str(row.get(name_field, f"bair_{idx}")),
            "area_km2": round(metrics["area_km2"], 4),
            "paved_km": round(metrics["paved_km"], 3),
            "drain_km": round(metrics["drain_km"], 3),
            "canal_km": round(metrics["canal_km"], 3),
            "green_km2": round(metrics["green_km2"], 3),
            "pumps_n": int(metrics["pumps_n"]),
            "dens_pav_km_km2": ures["densities"]["dens_pav_km_km2"],
            "dreno_km_km2": ures["densities"]["dreno_km_km2"],
            "canal_km_km2": ures["densities"]["canal_km_km2"],
            "frac_verde": ures["densities"]["frac_verde"],
            "u_cobertura": ures["subindices"]["u_cobertura"],
            "u_micro": ures["subindices"]["u_micro"],
            "u_macro": ures["subindices"]["u_macro"],
            "u_permeabilidade": ures["subindices"]["u_permeabilidade"],
            "U_static": ures["U_static"],
            "dryness_date": ures["dynamic"]["calc_date"],
            "sm_norm": ures["dynamic"]["sm_norm"],
            "et_scaled": ures["dynamic"]["et_scaled"],
            "dryness": ures["dynamic"]["dryness"],
            "U_t": ures["U_t"],
            "Fragilidade_t": ures["Fragilidade_t"],
        }
        rows.append(props)
        geoms.append(geom_wgs)

    out_gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")

    # 4) Exporta GeoJSON + CSV tabular
    out_geo = DATA_DIR / "canoas_bairros_u.geojson"
    out_csv = DATA_DIR / "canoas_bairros_u.csv"
    out_gdf.to_file(out_geo, driver="GeoJSON")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"OK: {len(out_gdf)} bairros.")
    print(f"- GeoJSON: {out_geo}")
    print(f"- CSV    : {out_csv}")

if __name__ == "__main__":
    main()
