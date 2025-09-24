from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import joblib

def train_and_evaluate_model(df):
    """
    Trains a RandomForestRegressor model and evaluates its performance.

    Args:
        df (pandas.DataFrame): The preprocessed dataframe with features.

    Returns:
        The trained model object.
    """
    print("\nStarting model training and evaluation...")

    # Define features (X) and target (y)
    # We will predict 'Global_active_power'
    features = ['hour', 'day_of_week', 'month', 'year', 'quarter',
                'Global_reactive_power', 'Voltage', 'Global_intensity',
                'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    target = 'Global_active_power'

    X = df[features]
    y = df[target]

    # Split the data into training and testing sets
    # We'll use 80% for training and 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training data size: {X_train.shape[0]} samples")
    print(f"Testing data size: {X_test.shape[0]} samples")

    # Initialize and train the model
    # Using RandomForest as it's powerful and less prone to overfitting
    model = RandomForestRegressor(
        n_estimators=100,  # Number of trees in the forest
        random_state=42,
        n_jobs=-1,         # Use all available CPU cores
        max_features='sqrt', # The number of features to consider when looking for the best split
        min_samples_leaf=5 # The minimum number of samples required to be at a leaf node
    )

    print("Training the RandomForestRegressor model... (This may take a few minutes)")
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("\n--- Model Evaluation ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print("------------------------")

    # Save the trained model to a file for future use
    model_filename = 'energy_consumption_model.pkl'
    joblib.dump(model, model_filename)
    print(f"Trained model saved to '{model_filename}'")

    return model
