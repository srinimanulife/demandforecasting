import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.api import ExponentialSmoothing, ARIMA
from prophet import Prophet
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

def croston_method(y):
    """
    Implements the Croston method for intermittent demand forecasting.
    """
    y_nz = y[y > 0]
    if len(y_nz) == 0:
        return np.zeros(len(y))
    
    # Time between non-zero demands
    intervals = np.diff(np.where(y > 0)[0])
    if len(intervals) == 0:
        intervals = [1]
        
    # Exponential smoothing on demand and intervals
    demand_model = ExponentialSmoothing(y_nz, initialization_method='estimated').fit()
    interval_model = ExponentialSmoothing(intervals, initialization_method='estimated').fit()
    
    forecast_demand = demand_model.forecast(1)[0]
    forecast_interval = interval_model.forecast(1)[0]
    
    if forecast_interval == 0: # Avoid division by zero
        return np.zeros(len(y))
        
    return forecast_demand / forecast_interval

def run_forecast(model_name, train, test, target_col='Total_units'):
    """
    A wrapper function to run different forecasting models.
    """
    y_train = train[target_col]
    y_test = test[target_col]
    
    predictions = []
    
    try:
        if model_name == 'ARIMA':
            model = ARIMA(y_train, order=(5,1,0)).fit()
            predictions = model.forecast(steps=len(y_test))
        
        elif model_name == 'ETS':
            model = ExponentialSmoothing(y_train, seasonal='add', seasonal_periods=7).fit()
            predictions = model.forecast(steps=len(y_test))
            
        elif model_name == 'Prophet':
            df_prophet = pd.DataFrame({'ds': train.index, 'y': y_train})
            model = Prophet()
            model.fit(df_prophet)
            future = model.make_future_dataframe(periods=len(y_test))
            forecast = model.predict(future)
            predictions = forecast['yhat'][-len(y_test):].values
            
        elif model_name == 'Croston':
            # Croston's method gives a single value forecast
            forecast_value = croston_method(y_train.values)
            predictions = np.full(len(y_test), forecast_value)

        elif model_name == 'Zero-Inflated':
            # Classifier to predict zero vs. non-zero
            clf = LogisticRegression()
            y_train_class = (y_train > 0).astype(int)
            clf.fit(pd.DataFrame(y_train.index.dayofyear), y_train_class)
            
            # Regressor for non-zero values
            reg = RandomForestRegressor()
            y_train_reg = y_train[y_train > 0]
            if len(y_train_reg) > 0:
                reg.fit(pd.DataFrame(y_train_reg.index.dayofyear), y_train_reg)
            
            class_preds = clf.predict(pd.DataFrame(y_test.index.dayofyear))
            
            reg_preds = []
            if len(y_train_reg) > 0:
                reg_preds = reg.predict(pd.DataFrame(y_test.index.dayofyear))
            else:
                reg_preds = np.zeros(len(y_test))

            predictions = class_preds * reg_preds

        else: # ML models
            def create_features(df):
                df['dayofyear'] = df.index.dayofyear
                df['dayofweek'] = df.index.dayofweek
                df['month'] = df.index.month
                return df

            train_feat = create_features(train.copy())
            test_feat = create_features(test.copy())
            
            features = ['dayofyear', 'dayofweek', 'month']
            X_train, y_train_ml = train_feat[features], train_feat[target_col]
            X_test = test_feat[features]

            if model_name == 'LightGBM':
                model = lgb.LGBMRegressor(random_state=42)
                model.fit(X_train, y_train_ml)
                predictions = model.predict(X_test)
            
            elif model_name == 'RandomForest':
                model = RandomForestRegressor(random_state=42)
                model.fit(X_train, y_train_ml)
                predictions = model.predict(X_test)

    except Exception as e:
        print(f"Error running {model_name}: {e}")
        predictions = np.zeros(len(y_test)) # Return zeros on failure

    # Ensure predictions are non-negative
    predictions[predictions < 0] = 0
    
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    
    return predictions, {'RMSE': rmse, 'MAE': mae}
