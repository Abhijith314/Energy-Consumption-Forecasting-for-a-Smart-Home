import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import TensorFlow/Keras for LSTM
# Use a try-except block for graceful error handling if tensorflow isn't installed
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input
    TENSORFLOW_INSTALLED = True
except ImportError:
    TENSORFLOW_INSTALLED = False

def create_lstm_dataset(X, y, time_steps=1):
    """
    Reshapes data for LSTM model (samples, time_steps, features).
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def run_model_comparison(df):
    """
    Trains and evaluates all models on the same, fair sample of data.
    """
    print("Step 3: Starting model training and comparison...")

    # --- 1. Define Features (X) and Target (y) ---
    features = ['hour', 'day_of_week', 'month', 'year', 'quarter', 
                'Global_reactive_power', 'Voltage', 'Global_intensity', 
                'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    target = 'Global_active_power'

    X = df[features]
    y = df[target]

    # --- 2. Split Data ---
    # We split the data first to ensure our test set is consistent
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 3. Create a Fair Sample for All Models ---
    # We will use the same sample size for all models to ensure a fair comparison
    # This is necessary because SVR and LSTMs are too slow on the full dataset
    SAMPLE_SIZE = 10000 
    
    # We create a random sample from the training data
    # We use 'random_state' to ensure the sample is the same every time
    X_train_sample = X_train.sample(n=SAMPLE_SIZE, random_state=42)
    y_train_sample = y_train.loc[X_train_sample.index]

    print(f"--- All models will be trained on a fair sample of {SAMPLE_SIZE} data points ---")

    # --- 4. Scaling ---
    # Scale data for SVR and LSTM models
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_sample)
    X_test_scaled = scaler_X.transform(X_test)
    
    # --- 5. Model Training and Evaluation ---
    model_results = {} # To store results for the final report

    # --- Model 1: RandomForest Regressor ---
    print("\nTraining RandomForest Regressor...")
    start_time = time.time()
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_sample, y_train_sample)
    rf_time = time.time() - start_time
    rf_preds = rf_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    model_results['RandomForest'] = {'MAE': rf_mae, 'RMSE': rf_rmse, 'Time': rf_time}
    print(f"RandomForest took {rf_time:.2f} seconds.")
    print("--- RandomForest Evaluation ---")
    print(f"Mean Absolute Error (MAE): {rf_mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rf_rmse:.4f}")
    print("-" * 36)

    # --- Model 2: XGBoost Regressor ---
    print("\nTraining XGBoost Regressor...")
    start_time = time.time()
    xgb_model = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train_sample, y_train_sample)
    xgb_time = time.time() - start_time
    xgb_preds = xgb_model.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_preds)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
    model_results['XGBoost'] = {'MAE': xgb_mae, 'RMSE': xgb_rmse, 'Time': xgb_time}
    print(f"XGBoost took {xgb_time:.2f} seconds.")
    print("--- XGBoost Evaluation ---")
    print(f"Mean Absolute Error (MAE): {xgb_mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {xgb_rmse:.4f}")
    print("-" * 31)

    # --- Model 3: Support Vector Regressor (SVR) ---
    print("\nTraining Support Vector Regressor...")
    start_time = time.time()
    svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    # SVR uses the scaled data
    svr_model.fit(X_train_scaled, y_train_sample) 
    svr_time = time.time() - start_time
    svr_preds = svr_model.predict(X_test_scaled)
    svr_mae = mean_absolute_error(y_test, svr_preds)
    svr_rmse = np.sqrt(mean_squared_error(y_test, svr_preds))
    model_results['SVR'] = {'MAE': svr_mae, 'RMSE': svr_rmse, 'Time': svr_time}
    print(f"SVR took {svr_time:.2f} seconds.")
    print("--- SVR Evaluation ---")
    print(f"Mean Absolute Error (MAE): {svr_mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {svr_rmse:.4f}")
    print("-" * 27)

    # --- Model 4: LSTM ---
    if TENSORFLOW_INSTALLED:
        print("\nTraining LSTM model...")
        start_time = time.time()
        
        # Reshape data for LSTM
        # We need to reshape the scaled training data
        TIME_STEPS = 1 # Using 1 time step for a simple model
        n_features = X_train_scaled.shape[1]
        
        # Create a new DataFrame for easier reshaping
        X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train_sample.index)
        y_train_sample_df = pd.Series(y_train_sample, index=X_train_sample.index)
        
        X_lstm, y_lstm = create_lstm_dataset(X_train_scaled_df, y_train_sample_df, TIME_STEPS)
        
        # Build the LSTM model
        lstm_model = Sequential()
        # Use new Keras 3.0 Input layer
        lstm_model.add(Input(shape=(TIME_STEPS, n_features))) 
        lstm_model.add(LSTM(50, activation='relu'))
        lstm_model.add(Dense(1))
        lstm_model.compile(optimizer='adam', loss='mse')

        # Train the model
        lstm_model.fit(X_lstm, y_lstm, epochs=10, batch_size=32, verbose=1)
        lstm_time = time.time() - start_time

        # Prepare test data for LSTM prediction
        # We must scale and reshape the test data in the same way
        X_test_scaled_df = pd.DataFrame(X_test_scaled, index=X_test.index)
        y_test_df = pd.Series(y_test, index=X_test.index)
        X_test_lstm, y_test_lstm = create_lstm_dataset(X_test_scaled_df, y_test_df, TIME_STEPS)

        lstm_preds = lstm_model.predict(X_test_lstm)
        
        # We compare against y_test_lstm since some samples are lost during reshaping
        lstm_mae = mean_absolute_error(y_test_lstm, lstm_preds)
        lstm_rmse = np.sqrt(mean_squared_error(y_test_lstm, lstm_preds))
        model_results['LSTM'] = {'MAE': lstm_mae, 'RMSE': lstm_rmse, 'Time': lstm_time}
        print(f"LSTM took {lstm_time:.2f} seconds.")
        print("--- LSTM Evaluation ---")
        print(f"Mean Absolute Error (MAE): {lstm_mae:.4f}")
        print(f"Root Mean Squared Error (RMSE): {lstm_rmse:.4f}")
        print("-" * 27)
    else:
        print("\nSkipping LSTM model: TensorFlow/Keras not installed.")

    # --- 6. Final Report ---
    # Sort models by MAE (best to worst)
    sorted_results = sorted(model_results.items(), key=lambda item: item[1]['MAE'])

    print("\n\n" + "=" * 50)
    print("          FINAL MODEL COMPARISON REPORT")
    print(f"          (All models trained on {SAMPLE_SIZE} samples)")
    print("=" * 50)
    print("Model         \tMAE     \tRMSE    \tTraining Time (s)")
    print("-" * 50)
    for model_name, metrics in sorted_results:
        print(f"{model_name:<14}\t{metrics['MAE']:.6f}\t{metrics['RMSE']:.6f}\t{metrics['Time']:.6f}")
    print("\n* Lower MAE/RMSE is better.\n")

