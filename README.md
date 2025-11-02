
# Forecast.AI — API de Risco por Bairros

## Visão Geral
Este projeto expõe uma API FastAPI que calcula risco de inundação por bairro na cidade de Canoas/RS combinando dois pilares:

- **Perigo (H)** derivado de previsões do Open-Meteo (chuvas, probabilidade, umidade do solo, vazão fluvial).
- **Robustez/Infraestrutura (U)** calculada a partir de dados geoespaciais (GeoCanoas, OpenStreetMap) e indicadores de drenagem urbana.

O risco por bairro/data é publicado em múltiplos formatos (JSON, CSV e GeoJSON), com filtros avançados, ranking, detalhe por bairro e geração de insights via OpenAI (RAG com playbook local).

## Arquitetura
- `app.py`: serviço FastAPI principal (endpoints REST + integrações com OpenAI/Open-Meteo).
- `configs/weights.yaml`: pesos de agregação e limites de classificação (`green/yellow/red`).
- `services/`: scripts auxiliares para ingestão de dados e ETL.
  - `apimeteo_conn.py`: coleta dados meteorológicos/flood do Open-Meteo e gera `hazard_forecast.csv`.
  - `u_point_min.py`: compila indicadores de infraestrutura urbana (OSM + GeoCanoas) e sintetiza `U_t`.
  - `risk_by_bairro.py`: combinação offline de H e U para gerar camadas agregadas.
- `services/data/`: repositório de dados de entrada/saída (hazard, u, risk, cache de LLM).

## Estrutura do Repositório
```text
.
├── app.py
├── configs/
│   └── weights.yaml
├── services/
│   ├── apimeteo_conn.py
│   ├── risk_by_bairro.py
│   ├── u_point_min.py
│   └── data/
│       ├── hazard/
│       │   └── hazard_forecast.csv
│       ├── u/
│       │   ├── canoas_bairros_u.csv
│       │   └── canoas_bairros_u.geojson
│       ├── pop/
│       │   └── canoas_bairros_pop.csv (opcional)
│       ├── risk/
│       │   ├── canoas_bairros_risk.csv
│       │   └── canoas_bairros_risk.geojson
│       └── cache/
│           └── llm_insights.json
└── README.md
```

## Dados de Entrada
| Dataset | Local | Descrição | Colunas obrigatórias |
|---------|-------|-----------|----------------------|
| `hazard_forecast.csv` | `services/data/hazard/` | Previsões diárias de perigo (H_score) | `date`, `H_score` (+ opcionais `p6_pct`, `a72_pct`, `sm_norm`, `et_deficit`, `p1_pct`, `pp_unit`, `rd_norm`) |
| `canoas_bairros_u.csv` | `services/data/u/` | Indicadores de infraestrutura por bairro | `bairro`, `U_t` (ou `U_static`) e subíndices `u_cobertura`, `u_micro`, `u_macro`, `u_permeabilidade` |
| `canoas_bairros_u.geojson` | `services/data/u/` | Geometria e metadados dos bairros | `bairro`, propriedades usadas na API |
| `canoas_bairros_pop.csv` (opcional) | `services/data/pop/` | População por bairro para análises adicionais | `bairro`, `population` |

Os arquivos acima podem ser gerados via scripts auxiliares descritos adiante.

## Preparação dos Dados

### Hazard (H_score)
1. Ajuste `lat`, `lon` e horizontes em `services/apimeteo_conn.py` se necessário.
2. Execute o script a partir da raiz do projeto:
   ```bash
   python services/apimeteo_conn.py
   ```
3. O script consulta as APIs do Open-Meteo (weather + flood), agrega estatísticas diárias (p1, p6, probabilidade, umidade, evapotranspiração), normaliza e escreve `hazard_forecast.csv`.

### Infraestrutura/U (U_t)
1. Configure âncoras (`ANCHORS`) e pesos (`WEIGHTS`) conforme calibração local em `services/u_point_min.py`.
2. Execute:
   ```bash
   python services/u_point_min.py
   ```
3. O script baixa o GeoJSON oficial de bairros (GeoCanoas), consulta o Overpass API para métricas de pavimentação, drenagem, canalização, áreas verdes e bombas, normaliza cada indicador e computa U_static + U_t (com ajuste dinâmico de dryness via Open-Meteo).

### Camada de Risco (opcional offline)
Para gerar um CSV/GeoJSON estático com todas as combinações H×U:
```bash
python services/risk_by_bairro.py
```
Esse passo não é obrigatório para servir a API, mas produz arquivos em `services/data/risk/` úteis para análises offline.

## Configurações
- Pesos de perigo (`hazard_daily_weights`) e robustez (`u_weights`) bem como limites de classificação (`hazard_levels`) residem em `configs/weights.yaml`.
- Ajuste os limites para calibrar clusters `green`, `yellow`, `red`.
- O arquivo é carregado dinamicamente pela API através de `load_weights()`.

## Dependências
Versão recomendada do Python: **3.11+** (necessário para pacotes geoespaciais recentes).

Instale as dependências principais em um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn[standard] python-dotenv pydantic pandas geopandas numpy pyyaml \
            openai openmeteo-requests requests-cache retry-requests shapely pyproj requests
```

Outros pacotes utilizados pelos scripts:
- `rich` (logs opcionais, não obrigatório).
- `tqdm` (para barras de progresso em ETLs longas).

No macOS/Linux pode ser necessário instalar bibliotecas do sistema para `geopandas`, `shapely` e `pyproj` (GDAL/GEOS/PROJ).

## Variáveis de Ambiente
Configure um arquivo `.env` (ou exporte variáveis) com os seguintes parâmetros:

| Variável | Descrição |
|----------|-----------|
| `OPENAI_API_KEY` | Obrigatória para geração de insights (endpoints `/v1/insights/*`). |
| `OPENAI_MODEL` | Opcional; padrão `gpt-4o-mini`. |
| `HTTP_PROXY` / `HTTPS_PROXY` | Opcional; suporte para ambientes com proxy corporativo. |

A API usa `python-dotenv` para carregar `.env` automaticamente no startup.

## Execução da API
1. Certifique-se de que os dados exigidos estejam presentes em `services/data/`.
2. Ative o ambiente virtual e inicie o servidor:
   ```bash
   uvicorn app:app --reload
   ```
3. Documentação interativa disponível em `http://127.0.0.1:8000/docs` (Swagger) ou `http://127.0.0.1:8000/redoc`.

### Exemplos de Consulta
- Listar risco de todos os bairros (data mais recente):
  ```bash
  curl 'http://127.0.0.1:8000/v1/risk/by_bairro'
  ```
- Top 5 bairros por risco em data específica:
  ```bash
  curl 'http://127.0.0.1:8000/v1/risk/by_bairro/top?date=2024-05-27&n=5'
  ```
- GeoJSON para mapas:
  ```bash
  curl 'http://127.0.0.1:8000/v1/geo/canoas/bairros_risk?include=all' \
       -o canoas_risk.geojson
  ```

## Endpoints Principais
| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/health` | Verificação simples de vida. |
| `GET` | `/v1/meta` | Metadados (datas disponíveis, pesos, thresholds). |
| `GET` | `/v1/bairros/list` | Lista bairros, status de dados e centróides (opcional). |
| `GET` | `/v1/risk/by_bairro` | Risco tabular com filtros por risco, subíndices e fatores de perigo. |
| `GET` | `/v1/risk/by_bairro/csv` | Exportação CSV do endpoint acima. |
| `GET` | `/v1/risk/by_bairro/top` | Ranking Top-N por data. |
| `GET` | `/v1/geo/canoas/bairros_risk` | GeoJSON para visualização em mapas. |
| `GET` | `/v1/bairros/detail` | Detalhe completo de um bairro (U dinâmico opcional). |
| `GET` | `/v1/filters` | Esquema de filtros para front-ends. |
| `GET` | `/v1/insights/by_bairro` | Insight textual (RAG) por bairro/data; usa cache local. |
| `GET` | `/v1/insights/city_top` | Síntese operacional municipal dos Top-N bairros. |

## Cache e Persistência
- `services/data/cache/llm_insights.json`: cache JSON com as respostas da OpenAI para cada bairro/data. Evita chamadas repetidas; é versionado localmente.
- Para limpar o cache basta remover o arquivo:
  ```bash
  rm services/data/cache/llm_insights.json
  ```

## Boas Práticas Operacionais
- Automatize a coleta de hazard (`apimeteo_conn.py`) duas vezes por dia (cron ou Airflow) e publique o CSV.
- Regere `U_t` periodicamente (mensalmente ou após grandes obras). O script `u_point_min.py` pode demorar devido à coleta do Overpass API — utilize cache (`requests_cache`) para suavizar.
- Versione os arquivos de dados historicamente para rastrear regressões.
- Em produção, execute o FastAPI com um servidor ASGI robusto (ex.: `uvicorn --workers 4` ou `gunicorn -k uvicorn.workers.UvicornWorker`).

## Próximos Passos
- Expandir módulos de ingestão para outras cidades (parametrizar pesos e âncoras).
- Integrar testes automatizados para os pipelines de ETL e validações de schema de dados.
- Publicar imagens Docker com os scripts de pré-processamento e a API para facilitar deploys portáteis.
