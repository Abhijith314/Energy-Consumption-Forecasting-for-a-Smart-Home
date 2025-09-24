# Energy Consumption Forecasting ⚡️

This project forecasts household energy consumption using a machine learning model. The model is trained on historical data to predict future energy usage. This README will guide you through setting up and running the project.

## About The Project 📖

This project uses a Random Forest Regressor model to predict the global active power consumption of a household. The pipeline includes data preprocessing, exploratory data analysis (EDA), and model training. The final trained model is saved as `energy_consumption_model.pkl`.

### Built With

  * [pandas](https://pandas.pydata.org/)
  * [scikit-learn](https://scikit-learn.org/stable/)
  * [matplotlib](https://matplotlib.org/)
  * [seaborn](https://seaborn.pydata.org/)
  * [joblib](https://joblib.readthedocs.io/en/latest/)
  * [numpy](https://numpy.org/)

## Getting Started 🚀

To get a local copy up and running follow these simple steps.

### Prerequisites

Make sure you have Python 3 installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

### Installation

1.  **Clone the repo**
    ```sh
    git clone https://github.com/your_username_/your_project_name.git
    ```
2.  **Navigate to the project directory**
    ```sh
    cd your_project_name
    ```
3.  **Install the required packages**
    ```sh
    pip install -r requirements.txt
    ```

## Usage 🏃‍♀️

To run the project, you first need to have the dataset in the project's root directory. See the [Dataset](https://www.google.com/search?q=%23dataset) section for instructions on how to get the data.

Once the dataset is in place, run the main script:

```sh
python main.py
```

This will execute the entire pipeline:

1.  Load and preprocess the data.
2.  Perform and display exploratory data analysis plots.
3.  Train a Random Forest Regressor model and save it as `energy_consumption_model.pkl`.
4.  Print the model's evaluation metrics.

## Project Structure 📂

Here's an overview of the project's structure:

```
├── data_preprocessing.py
├── exploratory_data_analysis.py
├── main.py
├── model_training.py
├── requirements.txt
└── household_power_consumption.txt  <-- You need to add this file
```

  * `main.py`: The main script that runs the entire pipeline.
  * `data_preprocessing.py`: Contains the function to load and preprocess the data.
  * `exploratory_data_analysis.py`: Contains the function to perform and display EDA plots.
  * `model_training.py`: Contains the function to train and evaluate the machine learning model.
  * `requirements.txt`: A list of all the Python libraries required for this project.
  * `household_power_consumption.txt`: The dataset file (you need to download this).

## Workflow 📝

The project follows these steps:

1.  **Data Preprocessing**:

      * The `load_and_preprocess_data` function in `data_preprocessing.py` loads the `household_power_consumption.txt` file.
      * It cleans the data by handling missing values and converting data types.
      * It performs feature engineering to extract time-based features like hour, day of the week, month, year, and quarter.

2.  **Exploratory Data Analysis (EDA)**:

      * The `perform_eda` function in `exploratory_data_analysis.py` generates visualizations to understand the data.
      * It creates plots for:
          * Total daily global active power consumption over time.
          * Power consumption by the hour of the day.
          * Power consumption by the day of the week.

3.  **Model Training and Evaluation**:

      * The `train_and_evaluate_model` function in `model_training.py` trains the machine learning model.
      * It splits the data into training and testing sets.
      * It uses a `RandomForestRegressor` to learn from the features.
      * The model is evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
      * The trained model is saved to `energy_consumption_model.pkl`.

## Dataset 📊

The primary input for the project is a single data file containing historical power consumption data from a household. The code is specifically written to work with the **"Individual household electric power consumption"** dataset.

### How to Get the Input File (Step-by-Step)

1.  **Go to the UCI Dataset Webpage**
    Open your web browser and navigate to this URL: [https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption)

2.  **Download the Data**
    On the webpage, look for the "Data Folder" link and click on it. This will take you to a directory where you will find a file named `household_power_consumption.zip`. Click on this file to download it.

3.  **Unzip the File**
    Once the download is complete, unzip the `household_power_consumption.zip` file. Inside, you will find the `household_power_consumption.txt` file.

4.  **Place the File in Your Project Directory**
    For the Python code to find the file, you must place `household_power_consumption.txt` in the same folder as your Python scripts (`main.py`, `data_preprocessing.py`, etc.).

-----
