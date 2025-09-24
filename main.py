# Import the functions from your other modules
from data_preprocessing import load_and_preprocess_data
from exploratory_data_analysis import perform_eda
from model_training import train_and_evaluate_model

def main():
    """
    Main function to run the entire energy forecasting pipeline.
    """
    # Define the path to your dataset
    # Make sure 'household_power_consumption.txt' is in the same directory
    # or provide the full path.
    data_file_path = 'household_power_consumption.txt'

    # Step 1: Load and preprocess the data
    # We will use a smaller fraction of the data for a quicker demonstration
    # For a full analysis, you can process the entire dataset
    processed_df = load_and_preprocess_data(data_file_path)

    # Let's resample to hourly to make the dataset smaller and faster to process
    # For a more accurate model, you might use the original minute-level data
    # or resample differently.
    processed_df_hourly = processed_df.resample('h').mean()
    processed_df_hourly.dropna(inplace=True) # Drop hours with no data

    # Step 2: Perform Exploratory Data Analysis
    perform_eda(processed_df_hourly)

    # Step 3: Train and evaluate the machine learning model
    trained_model = train_and_evaluate_model(processed_df_hourly)

    print("\nProject pipeline finished successfully!")

if __name__ == '__main__':
    main()
