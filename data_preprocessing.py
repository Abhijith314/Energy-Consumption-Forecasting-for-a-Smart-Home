import pandas as pd

def load_and_preprocess_data(file_path):
    """
    Loads the household power consumption data, preprocesses it, and prepares it for analysis.

    Args:
        file_path (str): The path to the CSV data file.

    Returns:
        pandas.DataFrame: A preprocessed DataFrame ready for modeling.
    """
    # Define column names as the file doesn't have a header
    col_names = ['Date', 'Time', 'Global_active_power', 'Global_reactive_power', 'Voltage',
                 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

    # Load the data, treating '?' as missing values
    df = pd.read_csv(
        file_path,
        sep=';',
        header=0,
        names=col_names,
        low_memory=False,
        na_values=['?'],
        infer_datetime_format=True,
        parse_dates={'datetime': ['Date', 'Time']}
    )

    # --- Data Cleaning ---
    # Drop rows with missing values
    df.dropna(inplace=True)

    # Convert columns to appropriate numeric types
    for col in df.columns:
        if col != 'datetime':
            df[col] = pd.to_numeric(df[col])

    # Set the datetime column as the index
    df.set_index('datetime', inplace=True)

    # --- Feature Engineering ---
    # Create time-based features that the model can learn from
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek  # Monday=0, Sunday=6
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['quarter'] = df.index.quarter

    print("Data loading and preprocessing complete.")
    print(f"Data shape: {df.shape}")
    print("First 5 rows of the processed data:")
    print(df.head())

    return df
