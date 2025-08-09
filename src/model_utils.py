# small helper utilities for model loading/prediction
import joblib, os
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_california_housing

def load_or_train_default(path='models/best_model.pkl'):
    if os.path.exists(path):
        return joblib.load(path)
    # train a tiny default model so the API and tests can run
    df = fetch_california_housing(as_frame=True).frame
    X = df.drop(columns=['MedHouseVal'])
    y = df['MedHouseVal']
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    pipeline = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
    pipeline.fit(X, y)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    return pipeline
