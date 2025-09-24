import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df):
    """
    Performs and displays exploratory data analysis on the dataframe.

    Args:
        df (pandas.DataFrame): The preprocessed dataframe.
    """
    print("\nStarting Exploratory Data Analysis (EDA)...")

    # Resample to daily frequency for clearer long-term plots
    df_daily = df['Global_active_power'].resample('D').sum()

    # 1. Plot Global Active Power over time
    plt.figure(figsize=(15, 7))
    df_daily.plot(title='Total Daily Global Active Power Consumption')
    plt.xlabel('Date')
    plt.ylabel('Global Active Power (kilowatts)')
    plt.grid(True)
    plt.show()

    # 2. Plot average power consumption by hour of the day
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='hour', y='Global_active_power', data=df)
    plt.title('Power Consumption by Hour of the Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Global Active Power (kilowatts)')
    plt.grid(True)
    plt.show()

    # 3. Plot average power consumption by day of the week
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='day_of_week', y='Global_active_power', data=df)
    plt.title('Power Consumption by Day of the Week')
    plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
    plt.ylabel('Global Active Power (kilowatts)')
    plt.grid(True)
    plt.show()

    print("EDA plots have been generated.")
