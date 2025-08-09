import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

def load_data():
    # prefer local CSV if present (DVC or manual), else fetch from sklearn
    path = 'data/raw/housing.csv'
    if os.path.exists(path):
        import pandas as pd
        df = pd.read_csv(path)
        return df
    ds = fetch_california_housing(as_frame=True)
    return ds.frame

def make_pipeline(model):
    return Pipeline([('scaler', StandardScaler()), ('model', model)])

def main():
    df = load_data()
    X = df.drop(columns=['MedHouseVal'])
    y = df['MedHouseVal']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42)
    }

    mlflow.set_experiment('housing_experiment')
    best_rmse = float('inf')
    best_name = None

    for name, mdl in models.items():
        with mlflow.start_run(run_name=name) as run:
            pipeline = make_pipeline(mdl)
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))

            mlflow.log_param('model', name)
            mlflow.log_metric('rmse', float(rmse))
            mlflow.sklearn.log_model(pipeline, 'model')

            print(f"Run {name} - RMSE: {rmse:.4f}")
            if rmse < best_rmse:
                best_rmse = rmse
                best_name = name

    # retrain best on full data and save locally
    final_pipe = make_pipeline(models[best_name])
    final_pipe.fit(X, y)
    os.makedirs('models', exist_ok=True)
    joblib.dump(final_pipe, 'models/best_model.pkl')
    print('Saved best model:', best_name)

if __name__ == '__main__':
    main()
