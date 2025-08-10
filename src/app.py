# FastAPI + request/response helpers
from fastapi import FastAPI, Request, Query
from fastapi.responses import PlainTextResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles        # only needed if you use static files/templates
from fastapi.templating import Jinja2Templates     # only needed if you use templates
from pydantic import BaseModel
import joblib, logging, os
# Prometheus pieces
from prometheus_client import Counter, Gauge, Histogram, Summary, generate_latest, CONTENT_TYPE_LATEST

# Local helper to load/train the model (ensure src/model_utils.py exists)
from .model_utils import load_or_train_default

from html import escape
from urllib.parse import urlencode
from typing import Optional
import math
import sqlite3
import time

# ------------------ logging ------------------
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

# Load model (or train a default one if missing)
model = load_or_train_default('models/best_model.pkl')

# Prometheus metrics
# Request counters (labels: endpoint, method, status)
REQUESTS = Counter('app_requests_total',
                   'Total HTTP requests',
                   ['endpoint', 'method', 'status'])
# Request latency histogram (seconds)
REQUEST_LATENCY = Histogram('app_request_latency_seconds',
                            'Request latency in seconds',
                            ['endpoint'])
# Prediction value histogram (useful for dashboards)
PRED_HIST = Histogram('app_prediction_values',
                      'Histogram of prediction values')

# Prediction summary for quantiles (optional)
PRED_SUMMARY = Summary('app_prediction_summary',
                       'Summary (quantiles) of predictions')

# Last prediction (gauge)
LAST_PRED = Gauge('app_last_prediction', 'Last prediction value')

# Total predictions counter
PRED_COUNT = Counter('app_prediction_count_total', 'Total number of predictions made')

# Prediction errors
PRED_ERRORS = Counter('app_prediction_errors_total', 'Total prediction errors')

# DB inserts counter
DB_INSERTS = Counter('app_db_inserts_total', 'Total DB inserts into preds table')

# Gauge for number of rows in SQLite logs
SQLITE_ROWS = Gauge('app_sqlite_rows', 'Number of rows in predictions.sqlite DB')

# Model info: labels for model name & version, value always 1
# e.g., app_model_info{model="RandomForest",version="v1"}
MODEL_INFO = Gauge('app_model_info', 'Model metadata (label gauge)', ['model', 'version'])

# set model metadata (change labels as needed)
MODEL_INFO.labels(model='housing_model', version='v1').set(1)

app = FastAPI(title='Housing Price Predictor')

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

def update_sqlite_rows_gauge():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS cnt FROM preds")
        cnt = cur.fetchone()['cnt'] if cur.fetchone() is None else cur.execute("SELECT COUNT(*) AS cnt FROM preds").fetchone()['cnt']
        # We'll re-run in a safe way:
        cur.execute("SELECT COUNT(*) AS cnt FROM preds")
        cnt = cur.fetchone()['cnt']
        SQLITE_ROWS.set(int(cnt))
        conn.close()
    except Exception as e:
        logger.error(f"Failed to update sqlite rows gauge: {e}")

def log_to_sqlite(input_obj, prediction):
    """
    Insert a row and update DB-related metrics.
    """
    ensure_table()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO preds (ts, input, prediction) VALUES (?,?,?)",
        (time.strftime("%Y-%m-%d %H:%M:%S"), str(input_obj), float(prediction))
    )
    conn.commit()
    conn.close()

    # Update counters/gauges
    DB_INSERTS.inc()
    # Update sqlite row count gauge (cheap for demo)
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS cnt FROM preds")
        cnt = cur.fetchone()['cnt']
        SQLITE_ROWS.set(int(cnt))
        conn.close()
    except Exception as e:
        logger.error(f"Error updating SQLITE_ROWS after insert demo: {e}")

# Initialize sqlite table and gauge on startup
ensure_table()
# set current count (safe)
try:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS cnt FROM preds")
    cnt = cur.fetchone()['cnt']
    SQLITE_ROWS.set(int(cnt))
    conn.close()
except Exception:
    SQLITE_ROWS.set(0)

# ------------------ Pydantic model ------------------

class InputData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# ------------------ Helper timing decorator ------------------
from functools import wraps

def instrument_endpoint(endpoint_name):
    """
    Decorator to instrument an endpoint with REQUESTS and REQUEST_LATENCY.
    Usage:
      @app.get('/...')
      @instrument_endpoint('endpoint_name')
      def ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            t0 = time.time()
            method = 'unknown'
            status = '500'
            try:
                # If Request passed in kwargs or args, infer method
                for a in args:
                    if hasattr(a, 'method'):
                        method = a.method
                if 'request' in kwargs:
                    method = kwargs['request'].method
                result = await func(*args, **kwargs) if callable(func) else func(*args, **kwargs)
                status = '200'
                return result
            except Exception as e:
                status = '500'
                PRED_ERRORS.inc()
                logger.exception("Endpoint error: %s", e)
                raise
            finally:
                elapsed = time.time() - t0
                REQUEST_LATENCY.labels(endpoint=endpoint_name).observe(elapsed)
                REQUESTS.labels(endpoint=endpoint_name, method=method, status=status).inc()
        return wrapper
    return decorator

# ------------------ Endpoints ------------------

@app.post('/predict')
async def predict(data: InputData):
    """
    /predict: make a prediction, log to sqlite, and export metrics.
    """
    endpoint = '/predict'
    t0 = time.time()
    try:
        # Prepare input vector
        x = [[
            data.MedInc, data.HouseAge, data.AveRooms, data.AveBedrms,
            data.Population, data.AveOccup, data.Latitude, data.Longitude
        ]]
        # Model predict
        pred = float(model.predict(x)[0])

        # Metrics updates
        LAST_PRED.set(pred)
        PRED_HIST.observe(pred)
        PRED_SUMMARY.observe(pred)
        PRED_COUNT.inc()

        # Log to sqlite (and DB metrics updated there)
        try:
            log_to_sqlite(x, pred)
        except Exception as db_e:
            logger.error(f"Failed to write to sqlite: {db_e}")

        # update request metrics
        elapsed = time.time() - t0
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
        REQUESTS.labels(endpoint=endpoint, method='POST', status='200').inc()

        logger.info(f"input={x} pred={pred}")
        return {'prediction': pred}
    except Exception as e:
        # update error counters
        PRED_ERRORS.inc()
        elapsed = time.time() - t0
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
        REQUESTS.labels(endpoint=endpoint, method='POST', status='500').inc()
        logger.exception("Prediction failed: %s", e)
        raise

@app.get('/metrics')
def metrics():
    """
    Prometheus scrape endpoint.
    """
    # Optionally refresh dynamic gauges here
    try:
        # update sqlite count gauge on each scrape (cheap for demo)
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS cnt FROM preds")
        cnt = cur.fetchone()['cnt']
        SQLITE_ROWS.set(int(cnt))
        conn.close()
    except Exception:
        pass

    # Return the aggregated metrics
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Inline HTML UI (no external files) retained — uses same metrics for /logs
@app.get("/logs/ui-inline", response_class=HTMLResponse)
async def logs_ui_inline(request: Request,
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

    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Total count
    count_q = f"SELECT COUNT(*) as cnt FROM preds {where_sql}"
    cur.execute(count_q, params)
    total = int(cur.fetchone()['cnt'])
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

    # Build base URL for links
    base = request.url.path

    def make_q(**kw):
        qp = {}
        qp.update({"page": page, "limit": limit})
        if min_pred is not None: qp["min_pred"] = min_pred
        if max_pred is not None: qp["max_pred"] = max_pred
        if search: qp["search"] = search
        qp.update(kw)
        return "?" + urlencode({k: v for k, v in qp.items() if v is not None})

    # Inline CSS
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

    rows_html = ""
    for r in rows:
        rid = r["id"]
        ts = escape(str(r["ts"]))
        inp = str(r["input"])
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
    # instrument metrics for this endpoint too
    REQUESTS.labels(endpoint='/logs/ui-inline', method='GET', status='200').inc()
    RETURN = HTMLResponse(content=html_content)
    return RETURN

@app.get('/logs')
async def get_logs(limit: int = 100, offset: int = 0):
    """
    Return recent logs from the SQLite 'preds' table (JSON).
    We instrument this endpoint as well.
    """
    endpoint = '/logs'
    t0 = time.time()
    ensure_table()
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    q = "SELECT rowid as id, ts, input, prediction FROM preds ORDER BY ts DESC LIMIT ? OFFSET ?"
    cur.execute(q, (limit, offset))
    rows = cur.fetchall()
    conn.close()

    elapsed = time.time() - t0
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
    REQUESTS.labels(endpoint=endpoint, method='GET', status='200').inc()
    return {"count": len(rows), "logs": [dict(r) for r in rows]}

@app.get('/logs/{log_id}')
async def get_log_by_id(log_id: int):
    endpoint = '/logs/{log_id}'
    t0 = time.time()
    ensure_table()
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    q = "SELECT rowid as id, ts, input, prediction FROM preds WHERE rowid = ?"
    cur.execute(q, (log_id,))
    row = cur.fetchone()
    conn.close()
    elapsed = time.time() - t0
    status = '200' if row else '404'
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
    REQUESTS.labels(endpoint=endpoint, method='GET', status=status).inc()
    if row is None:
        return {"error": "log not found", "id": log_id}
    return dict(row)