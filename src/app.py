# FastAPI + request/response helpers
from fastapi import FastAPI, Request, Query
from fastapi.responses import PlainTextResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles        # only needed if you use static files/templates
from fastapi.templating import Jinja2Templates     # only needed if you use templates
from pydantic import BaseModel
import joblib, logging, os
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
from .model_utils import load_or_train_default

from html import escape
from urllib.parse import urlencode
from typing import Optional
import math
import sqlite3
import time

# Logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

# Load model (or train a default one if missing)
model = load_or_train_default('models/best_model.pkl')

# Prometheus metrics
REQUESTS = Counter('app_requests_total', 'Total prediction requests')
LAST_PRED = Gauge('app_last_prediction', 'Last prediction value')

app = FastAPI(title='Housing Price Predictor')

class InputData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

def log_to_sqlite(input_dict, pred):
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS preds
                 (ts TEXT, input TEXT, prediction REAL)""")
    c.execute("INSERT INTO preds VALUES (?,?,?)", (time.strftime('%Y-%m-%d %H:%M:%S'), str(input_dict), float(pred)))
    conn.commit()
    conn.close()

@app.post('/predict')
def predict(data: InputData):
    REQUESTS.inc()
    x = [[
        data.MedInc, data.HouseAge, data.AveRooms, data.AveBedrms,
        data.Population, data.AveOccup, data.Latitude, data.Longitude
    ]]
    pred = model.predict(x)[0]
    LAST_PRED.set(float(pred))
    logger.info(f"input={x} pred={pred}")
    try:
        log_to_sqlite(x, pred)
    except Exception as e:
        logger.error(f"Failed to log to sqlite: {e}")
    return {'prediction': float(pred)}

@app.get('/metrics')
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/logs/ui-inline", response_class=HTMLResponse)
def logs_ui_inline(request: Request,
                   page: int = 1,
                   limit: int = 20,
                   min_pred: Optional[float] = None,
                   max_pred: Optional[float] = None,
                   search: Optional[str] = None):
    """
    Return a fully self-contained HTML page showing logs from SQLite.
    No external files required. Supports pagination, min/max prediction filters and simple search.
    """
    ensure_table()

    # Build WHERE clauses safely
    where_clauses = []
    params = []
    if min_pred is not None:
        where_clauses.append("prediction >= ?")
        params.append(min_pred)
    if max_pred is not None:
        where_clauses.append("prediction <= ?")
        params.append(max_pred)
    if search:
        where_clauses.append("input LIKE ?")
        params.append(f"%{search}%")

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    conn = sqlite3.connect("predictions.db", timeout=10)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Total count
    count_q = f"SELECT COUNT(*) as cnt FROM preds {where_sql}"
    cur.execute(count_q, params)
    total = cur.fetchone()["cnt"] if cur.fetchone() is None else cur.execute(count_q, params).fetchone()["cnt"]
    # (defensive) but we'll re-run properly:
    cur.execute(count_q, params)
    total = int(cur.fetchone()["cnt"])

    total_pages = max(1, math.ceil(total / limit))
    if page < 1:
        page = 1
    if page > total_pages:
        page = total_pages

    offset = (page - 1) * limit

    q = f"SELECT rowid as id, ts, input, prediction FROM preds {where_sql} ORDER BY ts DESC LIMIT ? OFFSET ?"
    row_params = params + [limit, offset]
    cur.execute(q, row_params)
    rows = cur.fetchall()
    conn.close()

    # Build base URL for links (path only)
    base = request.url.path

    def make_q(**kw):
        # merge current query params with kw overrides
        qp = {}
        qp.update({"page": page, "limit": limit})
        if min_pred is not None: qp["min_pred"] = min_pred
        if max_pred is not None: qp["max_pred"] = max_pred
        if search: qp["search"] = search
        qp.update(kw)
        return "?" + urlencode({k: v for k, v in qp.items() if v is not None})

    # Inline CSS (simple, elegant)
    css = """
    body { font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial; background: #f5f7fb; color: #172b4d; padding: 20px; }
    .wrap { max-width: 1100px; margin: 0 auto; }
    .header { display:flex; align-items:center; justify-content:space-between; margin-bottom:18px; }
    .card { background: white; border-radius:10px; padding:14px; box-shadow: 0 6px 18px rgba(23,43,77,0.06); }
    table { width:100%; border-collapse:collapse; font-size:0.95rem; }
    th, td { padding:10px 12px; text-align:left; border-bottom: 1px solid #eef2f6; }
    th { color:#536270; font-weight:600; font-size:0.85rem; }
    a { color:#0b5fff; text-decoration:none; }
    .muted { color:#7a8a97; font-size:0.9rem; }
    .controls { display:flex; gap:10px; align-items:end; flex-wrap:wrap; }
    .btn { background:#0b5fff; color:white; border-radius:8px; padding:8px 12px; display:inline-block; }
    .btn-light { background:#eef2f6; color:#172b4d; }
    .search { width:260px; padding:8px; border-radius:8px; border:1px solid #e1e8ef; }
    .pagebox { display:flex; gap:8px; align-items:center; justify-content:center; margin-top:12px; }
    .small { font-size:0.85rem; color:#69727b; }
    code.inline { background:#f4f7fb; padding:4px 6px; border-radius:6px; display:inline-block; max-width:60ch; white-space:pre-wrap; }
    """

    # Build rows html escaping inputs
    rows_html = ""
    for r in rows:
        rid = r["id"]
        ts = escape(str(r["ts"]))
        inp = str(r["input"])
        # truncate for display
        short = escape(inp if len(inp) <= 200 else inp[:200] + "...")
        pred = f"{float(r['prediction']):.4f}"
        rows_html += f"""
        <tr>
          <td><a href="/logs/{rid}">{rid}</a></td>
          <td>{ts}</td>
          <td><code class="inline">{short}</code></td>
          <td style="text-align:right"><strong>{pred}</strong></td>
        </tr>
        """

    if rows_html.strip() == "":
        rows_html = '<tr><td colspan="4" class="muted" style="text-align:center;padding:20px">No logs found.</td></tr>'

    # Pagination controls
    prev_disabled = "style='opacity:0.5;pointer-events:none'" if page <= 1 else ""
    next_disabled = "style='opacity:0.5;pointer-events:none'" if page >= total_pages else ""

    html_content = f"""<!doctype html>
    <html>
      <head>
        <meta charset="utf-8">
        <title>Logs — Housing Predictor</title>
        <meta name="viewport" content="width=device-width,initial-scale=1">
        <style>{css}</style>
      </head>
      <body>
        <div class="wrap">
          <div class="header">
            <div>
              <h2 style="margin:0">Prediction Logs</h2>
              <div class="muted small">Showing page {page} of {total_pages} — {total} total</div>
            </div>
            <div>
              <a class="btn" href="/docs">API Docs</a>
            </div>
          </div>

          <div class="card" style="margin-bottom:14px">
            <form method="get" action="{base}" style="display:flex;gap:10px;align-items:end;flex-wrap:wrap">
              <div style="display:flex;flex-direction:column">
                <label class="small muted">Min prediction</label>
                <input class="search" type="number" step="any" name="min_pred" value="{'' if min_pred is None else escape(str(min_pred))}">
              </div>
              <div style="display:flex;flex-direction:column">
                <label class="small muted">Max prediction</label>
                <input class="search" type="number" step="any" name="max_pred" value="{'' if max_pred is None else escape(str(max_pred))}">
              </div>
              <div style="display:flex;flex-direction:column">
                <label class="small muted">Search input</label>
                <input class="search" type="text" name="search" value="{'' if not search else escape(str(search))}">
              </div>
              <div style="display:flex;flex-direction:column">
                <label class="small muted">Per page</label>
                <select name="limit" style="padding:8px;border-radius:8px;border:1px solid #e1e8ef">
                  <option value="10" {'selected' if limit==10 else ''}>10</option>
                  <option value="20" {'selected' if limit==20 else ''}>20</option>
                  <option value="50" {'selected' if limit==50 else ''}>50</option>
                  <option value="100" {'selected' if limit==100 else ''}>100</option>
                </select>
              </div>
              <div style="display:flex;gap:8px">
                <button class="btn" type="submit">Apply</button>
                <a class="btn-light" href="{base}">Reset</a>
              </div>
            </form>
          </div>

          <div class="card">
            <div style="overflow-x:auto">
              <table>
                <thead>
                  <tr><th>ID</th><th>Timestamp</th><th>Input (truncated)</th><th style="text-align:right">Prediction</th></tr>
                </thead>
                <tbody>
                  {rows_html}
                </tbody>
              </table>
            </div>
          </div>

          <div class="pagebox">
            <a {prev_disabled} class="btn-light" href="{base}{make_q(page=page-1)}">Previous</a>
            <div class="small">Page {page} / {total_pages}</div>
            <a {next_disabled} class="btn-light" href="{base}{make_q(page=page+1)}">Next</a>
          </div>

          <div style="height:24px"></div>
          <div class="muted small">Note: this UI is served inline (no external files). For production, secure this route.</div>
        </div>
      </body>
    </html>"""

    return HTMLResponse(content=html_content)

def get_db_connection(db_path="predictions.db"):
    """
    Open a sqlite3 connection with Row factory. Use this for every request.
    """
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn

def ensure_table():
    """
    Create the preds table if it doesn't exist.
    Columns: ts (text), input (text), prediction (real)
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS preds (
            ts TEXT,
            input TEXT,
            prediction REAL
        )
    """)
    conn.commit()
    conn.close()