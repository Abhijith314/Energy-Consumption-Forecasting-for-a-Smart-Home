import pandas as pd

def load_and_preprocess_data(file_path):
    """
    Loads the household power consumption data, preprocesses it, and prepares it for analysis.

    Args:
        file_path (str): The path to the CSV data file.

    Returns:
        pandas.DataFrame: A preprocessed DataFrame ready for modeling.
    """
    print("Step 1: Loading and preprocessing data...")
    # Define column names as the file doesn't have a header
    col_names = ['Date', 'Time', 'Global_active_power', 'Global_reactive_power', 'Voltage',
                 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

    # Load the data, treating '?' as missing values and parsing dates
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
    df.dropna(inplace=True)

    # --- Data Type Conversion ---
    for col in df.columns:
        if col != 'datetime':
            df[col] = pd.to_numeric(df[col])

    # Set the datetime column as the index
    df.set_index('datetime', inplace=True)

    print("Data loading and preprocessing complete.")
    return df

