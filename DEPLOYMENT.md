# Deployment

This repository includes a Render Blueprint in `render.yaml` for the Streamlit dashboard and FastAPI scoring API.

## Before deployment

The checked-in processed D1–D4 artifacts are generated from the official retail dataset provided for Project FORESIGHT (deterministic 25,000-transaction selection, seed 42), processed through the official D1 pipeline. Forecasts, risk scores, decisions, and rupee values are model-derived planning outputs computed from that official data — they are not official Zidio or NorthBay Living business results. The dashboard and API must continue to display and document that distinction.

## Render deployment

1. Push the repository to a Git provider connected to Render.
2. In Render, create a new Blueprint and select this repository.
3. Confirm the two services from `render.yaml`:
   - `foresight-dashboard`: Streamlit on Render's `$PORT`
   - `foresight-scoring-api`: Uvicorn/FastAPI on Render's `$PORT`
4. Deploy both services.
5. Verify:
   - Dashboard URL loads the Streamlit app.
   - API URL `/health` returns `{"status":"ok", ...}`.
   - API URL `/data-status` reports the current data state.
   - API `POST /score/sku` returns forecast and risk blocks for a known SKU.
6. Record the public URLs in the submission form and README only after deployment succeeds.

No credentials, provider tokens, or public URLs are stored in this repository. Public deployment cannot be completed from the local workspace without access to the hosting account and repository integration.

## Local commands

```text
.venv\Scripts\python.exe -m streamlit run app\app.py
.venv\Scripts\python.exe -m uvicorn service.main:app --reload
```
