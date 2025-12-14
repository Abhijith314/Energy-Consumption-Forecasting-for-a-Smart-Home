# Import the functions from your other modules
from data_preprocessing import load_and_preprocess_data
from exploratory_data_analysis import perform_eda
from model_training import run_model_comparison

def main():
    """
    Main function to run the entire energy forecasting pipeline.
    """
    # Define the path to your dataset.
    # Make sure 'household_power_consumption.txt' is in the same directory.
    data_file_path = 'household_power_consumption.txt'

    # Step 1: Load and preprocess the data
    processed_df = load_and_preprocess_data(data_file_path)

    # Resample to hourly to make the dataset smaller and faster to process.
    processed_df_hourly = processed_df.resample('h').mean()
    processed_df_hourly.dropna(inplace=True)

    # Re-create time features after resampling
    processed_df_hourly['hour'] = processed_df_hourly.index.hour
    processed_df_hourly['day_of_week'] = processed_df_hourly.index.dayofweek
    processed_df_hourly['month'] = processed_df_hourly.index.month
    processed_df_hourly['year'] = processed_df_hourly.index.year
    processed_df_hourly['quarter'] = processed_df_hourly.index.quarter

    # Step 2: Perform Exploratory Data Analysis
    perform_eda(processed_df_hourly)

    # Step 3: Train, evaluate, and compare all machine learning models
    run_model_comparison(processed_df_hourly)

    print("\nProject pipeline finished successfully!")

if __name__ == '__main__':
    main()

