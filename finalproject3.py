import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import files # Import files for Colab upload
import io # Import io to read string data as a file

# --- 1. Data Collection (Now handles direct upload in Colab) ---

# --- 2. Data Loading & Exploration ---
def load_data_from_upload():
    """
    Load the COVID-19 dataset from a CSV file uploaded via Google Colab's files.upload().
    """
    print("Please upload your 'owid-covid-data.csv' file.")
    uploaded = files.upload() # This will open a file selection dialog

    file_name = None
    for fn in uploaded.keys():
        if fn.endswith('.csv'):
            file_name = fn
            break
    
    if file_name:
        try:
            # Read the uploaded CSV file into a pandas DataFrame
            # io.StringIO is used to treat the string content as a file
            df = pd.read_csv(io.StringIO(uploaded[file_name].decode('utf-8')))
            print(f"Data '{file_name}' loaded successfully!")
            return df
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
    else:
        print("No CSV file found among uploaded files. Please upload a CSV.")
        return None

def explore_data(df):
    """
    Perform basic exploration of the dataset.
    """
    print("\n--- Data Exploration ---")
    print("Columns in the dataset:")
    print(df.columns.tolist()) # Convert to list for cleaner output
    print("\nPreview of the data:")
    print(df.head())
    print("\nMissing values in each column:")
    print(df.isnull().sum())
    print("\nData Info:")
    df.info()

# --- 3. Data Cleaning ---
def clean_data(df):
    """
    Clean the dataset by filtering, handling missing values, and converting data types.
    """
    print("\n--- Data Cleaning ---")
    # Filter for countries of interest (Updated: USA, India, Canada)
    countries_of_interest = ['United States', 'India', 'Canada']
    df = df[df['location'].isin(countries_of_interest)].copy() # Use .copy() to avoid SettingWithCopyWarning
    
    # Convert date column to datetime before dropping NaNs to ensure proper sorting and interpolation
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date and location for correct interpolation
    df = df.sort_values(by=['location', 'date'])

    # Drop rows with missing critical values (total_cases, total_deaths, total_vaccinations)
    # total_vaccinations is crucial for vaccination analysis
    initial_rows = len(df)
    df.dropna(subset=['date', 'total_cases', 'total_deaths', 'total_vaccinations'], inplace=True)
    print(f"Dropped {initial_rows - len(df)} rows with missing critical values.")
    
    # Fill missing numeric values with interpolation (forward fill then backward fill)
    # This is better than just ffill for gaps
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
    
    print("Data cleaned successfully!")
    return df

# --- 4. Exploratory Data Analysis (EDA) & Visualizations ---

def plot_total_cases(df):
    """
    Plot total cases over time for selected countries.
    """
    print("\n--- Plotting Total COVID-19 Cases Over Time ---")
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df, x='date', y='total_cases', hue='location', marker='o', markersize=4)
    plt.title("Total COVID-19 Cases Over Time")
    plt.xlabel("Date")
    plt.ylabel("Total Cases")
    plt.legend(title='Country')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_total_deaths(df):
    """
    Plot total deaths over time for selected countries.
    """
    print("\n--- Plotting Total COVID-19 Deaths Over Time ---")
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df, x='date', y='total_deaths', hue='location', marker='o', markersize=4)
    plt.title("Total COVID-19 Deaths Over Time")
    plt.xlabel("Date")
    plt.ylabel("Total Deaths")
    plt.legend(title='Country')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def compare_daily_new_cases(df):
    """
    Compare daily new cases (smoothed) between selected countries.
    Using 'new_cases_smoothed' for better trend visualization.
    """
    print("\n--- Comparing Daily New COVID-19 Cases (Smoothed) ---")
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df, x='date', y='new_cases_smoothed', hue='location', marker='o', markersize=4)
    plt.title("Daily New COVID-19 Cases (Smoothed) Over Time")
    plt.xlabel("Date")
    plt.ylabel("New Cases (Smoothed)")
    plt.legend(title='Country')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def calculate_and_plot_death_rate(df):
    """
    Calculate and plot the death rate (total_deaths / total_cases) over time.
    """
    print("\n--- Calculating and Plotting Death Rate ---")
    # Avoid division by zero
    df['death_rate'] = (df['total_deaths'] / df['total_cases']) * 100
    # Replace inf with NaN and then fill NaNs (e.g., with 0 or previous valid value)
    df['death_rate'].replace([float('inf'), -float('inf')], pd.NA, inplace=True)
    df['death_rate'] = df['death_rate'].fillna(method='ffill').fillna(0) # Fill initial NaNs with 0 if no prior data

    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df, x='date', y='death_rate', hue='location', marker='o', markersize=4)
    plt.title("COVID-19 Death Rate (%) Over Time")
    plt.xlabel("Date")
    plt.ylabel("Death Rate (%)")
    plt.legend(title='Country')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# --- 5. Visualizing Vaccination Progress ---
def plot_cumulative_vaccinations(df):
    """
    Plot cumulative vaccinations over time for selected countries.
    """
    print("\n--- Plotting Cumulative Vaccinations Over Time ---")
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df, x='date', y='total_vaccinations', hue='location', marker='o', markersize=4)
    plt.title("Cumulative COVID-19 Vaccinations Over Time")
    plt.xlabel("Date")
    plt.ylabel("Total Vaccinations")
    plt.legend(title='Country')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def compare_vaccinated_population(df):
    """
    Compare the percentage of vaccinated population (people_vaccinated_per_hundred)
    for the latest available date for each country.
    """
    print("\n--- Comparing Vaccinated Population (%) ---")
    # Get the latest data for each country
    latest_data = df.loc[df.groupby('location')['date'].idxmax()]
    
    # Filter out countries with no vaccination data
    latest_data = latest_data.dropna(subset=['people_vaccinated_per_hundred'])

    if not latest_data.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=latest_data, x='location', y='people_vaccinated_per_hundred', palette='viridis')
        plt.title("Percentage of Population Vaccinated (Latest Data)")
        plt.xlabel("Country")
        plt.ylabel("People Vaccinated Per Hundred (%)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    else:
        print("No vaccination data available for comparison after cleaning.")

# --- Main Execution ---
if __name__ == "__main__":
    # Load the data using the Colab upload function
    data = load_data_from_upload()
    
    if data is not None:
        # Explore the data
        explore_data(data)
        
        # Clean the data
        cleaned_data = clean_data(data)
        
        # Perform EDA and Visualizations
        if not cleaned_data.empty:
            plot_total_cases(cleaned_data)
            plot_total_deaths(cleaned_data)
            compare_daily_new_cases(cleaned_data)
            calculate_and_plot_death_rate(cleaned_data)
            plot_cumulative_vaccinations(cleaned_data)
            compare_vaccinated_population(cleaned_data)
        else:
            print("\nCleaned data is empty. No plots can be generated.")
    
   
