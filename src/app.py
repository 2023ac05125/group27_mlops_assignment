from fastapi import FastAPI
from pydantic import BaseModel
import joblib, time, sqlite3, logging, os
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import PlainTextResponse
from .model_utils import load_or_train_default

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
