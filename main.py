import os
import pandas as pd
import numpy as np
import fastf1
import re
from glob import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind

def fetch_lap_data(years, save_dir="."):
    """
    Fetches lap data for Monaco GP for the given years and saves them as CSV files.

    Parameters:
        years (list of int): List of seasons/years to fetch Monaco GP lap data for.
        save_dir (str): Directory path to save the CSV files. Defaults to current directory.

    Returns:
        None

    Side Effects:
        - Creates one CSV file per year named 'monaco_laps_<year>.csv' in save_dir.
        - Prints progress and error messages to console.

    Example (successful fetch):
        >>> fetch_lap_data([2023], save_dir="test_data") # doctest: +ELLIPSIS
        Fetching lap data for Monaco 2023
        Saved lap data to test_data/monaco_laps_2023.csv

    Example (mixed results with valid and invalid years):
        >>> fetch_lap_data([1800, 2023, 2024], save_dir="test_data")  # doctest: +ELLIPSIS
        Fetching lap data for Monaco 1800
        Error fetching lap data for 1800: ...
        Fetching lap data for Monaco 2023
        Saved lap data to test_data/monaco_laps_2023.csv
        Fetching lap data for Monaco 2024
        Saved lap data to test_data/monaco_laps_2024.csv
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for year in years:
        try:
            print(f"Fetching lap data for Monaco {year}")
            session = fastf1.get_session(year, 'Monaco', 'Race')
            session.load(laps=True, telemetry=False, weather=False, messages=True)

            laps = session.laps
            laps['Year'] = year

            selected_columns = [
                'Time', 'Driver', 'DriverNumber', 'LapTime', 'LapNumber', 'Stint',
                'PitOutTime', 'PitInTime',
                'Sector1Time', 'Sector2Time', 'Sector3Time',
                'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime',
                'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST',
                'IsPersonalBest', 'Compound', 'TyreLife', 'FreshTyre',
                'Team', 'LapStartTime', 'LapStartDate', 'TrackStatus', 'Position',
                'Deleted', 'DeletedReason', 'FastF1Generated', 'IsAccurate',
                'Year'
            ]

            laps_cleaned = laps[selected_columns].dropna(subset=["LapTime"])
            file_name = os.path.join(save_dir, f"monaco_laps_{year}.csv")
            laps_cleaned.to_csv(file_name, index=False)
            print(f"Saved lap data to {file_name}")

        except Exception as e:
            print(f"Error fetching lap data for {year}: {e}")

def fetch_car_data(years, save_dir="."):
    """
    Fetches per-driver car telemetry data for Monaco GP for each specified year
    using the FastF1 API. Saves individual CSV files for each driver.

    Parameters:
        years (list of int): List of years (e.g., [2023, 2024]) to fetch telemetry data for.
        save_dir (str): Directory to save the output CSV files. Defaults to current directory.

    Returns:
        None

    Side Effects:
        - Creates CSV files named as 'car_data_<year>_driver_<driver_number>.csv' in save_dir.
        - Prints progress and error messages to stdout.

    Example (with one invalid and one valid year):
        >>> fetch_car_data([1800, 2023], save_dir="test_data")  # doctest: +ELLIPSIS
        Fetching car data for Monaco 1800
        Error loading session for 1800: ...
        Fetching car data for Monaco 2023
          Saved test_data/car_data_2023_driver_...
          ...
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for year in years:
        print(f"Fetching car data for Monaco {year}")
        try:
            session = fastf1.get_session(year, 'Monaco', 'Race')
            session.load(telemetry=True)

            for drv in session.drivers:
                try:
                    laps = session.laps.pick_drivers([drv])
                    driver_frames = []

                    for _, lap in laps.iterlaps():
                        tel = lap.get_car_data()
                        if tel.empty:
                            continue
                        tel['LapNumber'] = lap['LapNumber']
                        driver_frames.append(tel)

                    if not driver_frames:
                        print(f"  No telemetry found for driver {drv} in {year}")
                        continue

                    df_driver = pd.concat(driver_frames, ignore_index=True)
                    df_driver['DriverNumber'] = drv

                    # Reorder columns to match your original output
                    column_order = [
                        'Time', 'Date', 'RPM', 'nGear', 'Throttle', 'Brake', 'DRS', 'Source',
                        'LapNumber', 'DriverNumber'
                    ]
                    for col in column_order:
                        if col not in df_driver.columns:
                            df_driver[col] = pd.NA

                    df_driver = df_driver[column_order]

                    output_path = os.path.join(save_dir, f"car_data_{year}_driver_{drv}.csv")
                    df_driver.to_csv(output_path, index=False)
                    print(f"  Saved {output_path}")

                except Exception as e:
                    print(f"  Failed to process driver {drv} in {year}: {e}")

        except Exception as e:
            print(f"Error loading session for {year}: {e}")

def extract_driver_number(filename):
    """
    Extracts the driver number from a filename using a regular expression.

    This function assumes that the filename contains a segment like 'driver_XX',
    where XX is a numeric driver identifier.

    Parameters:
    ----------
    filename : str
        The filename string from which to extract the driver number.

    Returns:
    -------
    str or None
        The driver number as a string if found, otherwise None.

    Examples:
    --------
    >>> extract_driver_number("car_data_2021_driver_44.csv")
    '44'
    >>> extract_driver_number("car_data_driver_abc.csv") is None
    True
    >>> extract_driver_number("telemetry_2022_driver_7.csv")
    '7'
    """
    match = re.search(r'driver_(\d+)', filename)
    return match.group(1) if match else None

def aggregate_car_data(years, data_dir=".", output_prefix="aggregated_car_data"):
    """
    Aggregates per-driver car telemetry data (RPM, gear, throttle, brake, DRS)
    from individual CSV files for specified years. Saves per-year and combined
    aggregate CSVs.

    Parameters:
        years (list of int): List of years to aggregate telemetry for.
        data_dir (str): Directory containing the 'car_data_<year>_driver_<id>.csv' files.
        output_prefix (str): Prefix used for naming the output aggregated files.

    Returns:
        pd.DataFrame: Combined DataFrame with aggregated telemetry data for all years.
                      If no files found, returns empty DataFrame.

    Side Effects:
        - Saves per-year CSVs named as '<output_prefix>_<year>.csv'.
        - Saves a combined CSV named '<output_prefix>_all_years.csv'.
        - Prints progress and error messages to stdout.

    Doctest Example:
        >>> aggregate_car_data([1924, 2023], data_dir="test_data", output_prefix="test_output")  # doctest: +ELLIPSIS
        Aggregating car data for 1924
        No files found for 1924. Skipping
        Aggregating car data for 2023
        Saved: test_data/test_output_2023.csv
        Saved combined: test_data/test_output_all_years.csv
        ...
    """

    all_dfs = []

    for year in years:
        print(f"Aggregating car data for {year}")

        # Use glob to get all car_data files for this year
        file_pattern = os.path.join(data_dir, f"car_data_{year}_driver_*.csv")
        car_files = glob(file_pattern)

        if not car_files:
            print(f"No files found for {year}. Skipping")
            continue

        # Read and concatenate all files
        dfs = [pd.read_csv(f) for f in car_files]
        df = pd.concat(dfs, ignore_index=True)

        try:
            df['Brake'] = df['Brake'].astype(bool)
            df['LapNumber'] = df['LapNumber'].astype(int)
            df = df[df['nGear'] < 9]

            agg_df = df.groupby(['DriverNumber', 'LapNumber']).agg({
                'RPM': 'mean',
                'nGear': 'mean',
                'Throttle': 'mean',
                'Brake': lambda x: x.sum(),
                'DRS': lambda x: x.isin([10, 12, 14]).sum()
            }).reset_index()

            agg_df['Year'] = year
            output_file = os.path.join(data_dir, f"{output_prefix}_{year}.csv")
            agg_df.to_csv(output_file, index=False)
            all_dfs.append(agg_df)

            print(f"Saved: {output_file}")

        except Exception as e:
            print(f"Failed processing {year}: {e}")

    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        combined_file = os.path.join(data_dir, f"{output_prefix}_all_years.csv")
        df_all.to_csv(combined_file, index=False)
        print(f"Saved combined: {combined_file}")
        return df_all
    else:
        print("No data aggregated.")
        return pd.DataFrame()

def fetch_weather_data(years, save_dir="."):
    """
    Fetches weather data for the Monaco GP race session for the specified years
    using the FastF1 API and saves them as CSV files.

    Parameters:
        years (list of int): List of years to fetch weather data for.
        save_dir (str): Directory where the CSV files will be saved. Defaults to current directory.

    Returns:
        None

    Side Effects:
        - Saves weather data CSVs as 'weather_<year>.csv' in the specified directory.
        - Creates the save directory if it does not exist.
        - Prints progress and error messages to stdout.

    Doctest Example:
        >>> fetch_weather_data(["MMXX", 2023], save_dir="test_data")  # doctest: +ELLIPSIS
        Fetching weather data for Monaco MMXX
        Error fetching weather for MMXX: ...
        Fetching weather data for Monaco 2023
          Saved weather_2023.csv
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for year in years:
        try:
            print(f"Fetching weather data for Monaco {year}")
            session = fastf1.get_session(year, 'Monaco', 'Race')
            session.load(telemetry=False, weather=True)

            df_weather = session.weather_data.copy()
            df_weather.to_csv(os.path.join(save_dir, f"weather_{year}.csv"), index=False)
            print(f"  Saved weather_{year}.csv")
        except Exception as e:
            print(f"Error fetching weather for {year}: {e}")

def annotate_and_aggregate_weather(years, lap_data_dir=".", weather_dir=".", output_path="data/aggregated_weather_all_years.csv"):

    aggregated_weather_dfs = []

    for year in years:
        try:
            lap_file = os.path.join(lap_data_dir, f"monaco_laps_{year}.csv")
            weather_file = os.path.join(weather_dir, f"weather_{year}.csv")

            df_lap = pd.read_csv(lap_file)
            df_weather = pd.read_csv(weather_file)

            df_lap["Time"] = pd.to_timedelta(df_lap["Time"])
            df_weather["Time"] = pd.to_timedelta(df_weather["Time"])
            df_weather["LapNumber"] = None

            lap_boundaries = df_lap.groupby("LapNumber")["Time"].min().reset_index().sort_values("LapNumber")

            for i in range(len(lap_boundaries) - 1):
                lap_num = lap_boundaries.iloc[i]["LapNumber"]
                lap_start = lap_boundaries.iloc[i]["Time"]
                lap_end = lap_boundaries.iloc[i + 1]["Time"]
                mask = (df_weather["Time"] >= lap_start) & (df_weather["Time"] < lap_end)
                df_weather.loc[mask, "LapNumber"] = lap_num

            mask = df_weather["Time"] >= lap_boundaries.iloc[-1]["Time"]
            df_weather.loc[mask, "LapNumber"] = lap_boundaries.iloc[-1]["LapNumber"]

            df_weather = df_weather.dropna(subset=["LapNumber"])

            if "Time" in df_weather.columns:
                df_weather = df_weather.drop(columns=["Time"])

            agg_funcs = {col: 'mean' for col in df_weather.columns if col not in ['LapNumber', 'Rainfall']}
            if 'Rainfall' in df_weather.columns:
                agg_funcs['Rainfall'] = 'sum'

            df_agg = df_weather.groupby("LapNumber").agg(agg_funcs).reset_index()
            df_agg["Year"] = year
            aggregated_weather_dfs.append(df_agg)

            print(f"  Processed weather for {year}")

        except Exception as e:
            print(f"Could not process weather for {year}: {e}")

    df_all = pd.concat(aggregated_weather_dfs, ignore_index=True)
    df_all.to_csv(output_path, index=False)
    print(f" Aggregated weather saved to {output_path}")
    return df_all

def concat_lap_metadata(years, data_dir=".", output_path="lap_metadata_all_years.csv"):
    """
    Concatenates Monaco lap metadata CSVs across multiple years into one DataFrame,
    adds a 'Year' column to each, and saves the combined DataFrame as a new CSV.

    Parameters:
        years (list of int): List of years to include (must have corresponding 'monaco_laps_<year>.csv' files).
        data_dir (str): Directory where the lap CSVs are stored. Defaults to current directory.
        output_path (str): Output filename for the combined CSV. Relative to data_dir.

    Returns:
        pd.DataFrame: Combined DataFrame with all laps and a 'Year' column.

    Side Effects:
        - Saves the combined DataFrame to the specified output_path.
        - Prints a success message with file path.

    Doctest Example:
        >>> concat_lap_metadata([1746, 2021, 2023], data_dir="test_data", output_path="combined.csv")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        FileNotFoundError: [Errno 2] No such file or directory: 'test_data/monaco_laps_1746.csv'
    """
    lap_dfs = []
    for year in years:
        file_path = os.path.join(data_dir, f"monaco_laps_{year}.csv")
        df = pd.read_csv(file_path)
        df["Year"] = year
        lap_dfs.append(df)

    df_laps = pd.concat(lap_dfs, ignore_index=True)
    df_laps.to_csv(os.path.join(data_dir, output_path), index=False)
    print(f" Saved all-year lap metadata to {output_path}")
    return df_laps

def merge_car_with_laps(car_data_path, lap_metadata_df, output_path="car_lap_merged.csv"):
    """
    Merges aggregated telemetry data with full lap metadata.
    """
    df_car = pd.read_csv(car_data_path)
    merged = pd.merge(
        df_car,
        lap_metadata_df,
        on=['DriverNumber', 'LapNumber', 'Year'],
        how='right'
    )
    merged.to_csv(output_path, index=False)
    print(f" Merged car + lap data saved to {output_path}")
    return merged

def merge_with_weather(weather_data_path, car_lap_merged_df, output_path="merged_lap_car_weather_all_years.csv"):
    """
    Merges lap+car data with aggregated weather data.
    """
    df_weather = pd.read_csv(weather_data_path)
    final_df = pd.merge(
        df_weather,
        car_lap_merged_df,
        on=['LapNumber', 'Year'],
        how='right'
    )
    final_df.to_csv(output_path, index=False)
    print(f" Final full dataset saved to {output_path}")
    return final_df

def cleanup_intermediate_csvs(directory="data", keep_filename="merged_lap_car_weather_all_years.csv"):
    """
    Removes all CSV files in the given directory except the specified final dataset.
    """
    deleted_files = []

    for file in os.listdir(directory):
        if file.endswith(".csv") and file != keep_filename:
            try:
                os.remove(os.path.join(directory, file))
                deleted_files.append(file)
            except Exception as e:
                print(f"⚠️ Failed to delete {file}: {e}")

    print(f" Cleanup complete. Kept only: {keep_filename}")
    if deleted_files:
        print(f"Deleted {len(deleted_files)} files:")
        for f in deleted_files:
            print(f"  - {f}")

def add_simulated_weight_column(
    input_path="data/merged_lap_car_weather_all_years.csv",
    output_path="data/regression_with_weight.csv",
    total_laps=78,
    seed=42
):
    """
    Adds a simulated car weight column to a race dataset by estimating per-lap fuel burn
    and combining it with FIA-regulated minimum car weight per year.

    The weight is calculated for each row based on the lap number, a randomized
    starting and ending fuel amount per driver, and the known FIA base weights.

    Parameters:
        input_path (str): Path to the CSV file containing the race dataset.
                          Must include columns ['Year', 'DriverNumber', 'LapNumber'].
        output_path (str): Path where the updated CSV with the 'Weight' column will be saved.
        total_laps (int): Total number of laps in the race (default = 78).
        seed (int): Random seed for reproducibility of fuel burn simulations.

    Returns:
        pd.DataFrame: Modified DataFrame with an additional 'Weight' column.

    Side Effects:
        - Writes the updated DataFrame to the specified output_path.

    Doctest Examples:
        >>> import pandas as pd
        >>> test_df = pd.DataFrame({
        ...     'Year': [2024]*3,
        ...     'DriverNumber': [1, 1, 1],
        ...     'LapNumber': [1, 2, 3]
        ... })
        >>> test_df.to_csv("test_input.csv", index=False)
        >>> _ = add_simulated_weight_column("test_input.csv", "test_output.csv", total_laps=3, seed=0)
         Weight column added and saved to test_output.csv
        >>> df_out = pd.read_csv("test_output.csv")
        >>> len(df_out['Weight']) == len(df_out)
        True
        >>> isinstance(df_out['Weight'].median(), (int, float))  # median should be numeric
        True
    """
    np.random.seed(seed)

    # Load data
    df = pd.read_csv(input_path)

    # Minimum weight per FIA regulations
    min_weights = {
        2018: 734,
        2019: 743,
        2020: 746,
        2021: 752,
        2022: 795,
        2023: 798,
        2024: 798
    }

    # Cache for per-driver fuel strategy
    fuel_cache = {}

    def compute_weight(row):
        year = row['Year']
        driver = row['DriverNumber']
        lap = row['LapNumber']
        base_weight = min_weights[year]

        key = (year, driver)
        if key not in fuel_cache:
            initial_fuel = np.random.uniform(100, 110)
            final_fuel = np.random.uniform(1, 2)
            burn_per_lap = (initial_fuel - final_fuel) / (total_laps - 1)
            fuel_cache[key] = (initial_fuel, burn_per_lap)
        else:
            initial_fuel, burn_per_lap = fuel_cache[key]

        current_fuel = initial_fuel - (lap - 1) * burn_per_lap
        return base_weight + current_fuel

    # Apply function
    df['Weight'] = df.apply(compute_weight, axis=1)

    # Save
    df.to_csv(output_path, index=False)
    print(f" Weight column added and saved to {output_path}")
    return df

def preprocess_for_regression(
    input_path="data/regression_ready_with_weight.csv",
    output_path="data/regression_final.csv"
):
    """
    Cleans and preprocesses the merged F1 dataset to prepare it for regression modeling.
    Drops irrelevant columns, computes pit duration, encodes compounds, converts booleans,
    and outputs final regression-ready CSV.
    """
    df = pd.read_csv(input_path)

    # 1. Drop unnecessary columns
    columns_to_drop = [
        'AirTemp', 'Humidity', 'Pressure', 'WindDirection', 'WindSpeed', 'Rainfall',
        'Time', 'Driver', 'Sector1Time', 'Sector2Time', 'Sector3Time',
        'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime',
        'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST',
        'Team', 'LapStartTime', 'LapStartDate', 'Position',
        'Deleted', 'DeletedReason', 'FastF1Generated', 'IsAccurate'
    ]
    df.drop(columns=columns_to_drop, axis=1, inplace=True, errors='ignore')

    # 2. Initialize PitDuration with correct dtype
    df["PitDuration"] = pd.to_timedelta("NaT")
    df["PitInTime"] = pd.to_timedelta(df["PitInTime"])
    df["PitOutTime"] = pd.to_timedelta(df["PitOutTime"])

    # 3. Calculate PitDuration for each pit-in event
    for idx, row in df[df["PitInTime"].notna()].iterrows():
        mask = (
            (df["Year"] == row["Year"]) &
            (df["DriverNumber"] == row["DriverNumber"]) &
            (df["PitOutTime"].notna()) &
            (df.index > idx)
        )
        next_pit_out = df[mask].head(1)
        if not next_pit_out.empty:
            out_time = next_pit_out["PitOutTime"].iloc[0]  #  Correct fix
            df.at[idx, "PitDuration"] = out_time - row["PitInTime"]

    # 4. Encode compound types
    compound_order = {
        "HYPERSOFT": 1, "ULTRASOFT": 2, "SUPERSOFT": 3, "SOFT": 4,
        "MEDIUM": 5, "HARD": 6, "INTERMEDIATE": 7, "WET": 8
    }
    df["Compound"] = df["Compound"].map(compound_order)

    # 5. Drop columns no longer needed
    df.drop(['Year', 'DriverNumber', 'Stint', 'PitInTime', 'PitOutTime'], axis=1, inplace=True)

    # 6. Convert boolean columns to int
    df["FreshTyre"] = df["FreshTyre"].astype(int)
    df["IsPersonalBest"] = df["IsPersonalBest"].astype(int)

    # 7. Convert time columns to seconds
    df["LapTime"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()
    df["PitDuration"] = pd.to_timedelta(df["PitDuration"]).dt.total_seconds()

    # 8. Save result
    df.to_csv(output_path, index=False)
    print(f" Final regression dataset saved to {output_path}")
    return df

def train_random_forest(df):
    """
    Train a Random Forest Regressor to predict LapTime from telemetry and weather features.

    Args:
        df (pd.DataFrame): Preprocessed regression dataset. Must contain the following columns:
            - 'LapNumber', 'LapTime', 'FreshTyre', 'IsPersonalBest'
            - All model input features (e.g., 'Throttle', 'Brake', 'TrackTemp', etc.)

    Returns:
        model (RandomForestRegressor): Trained scikit-learn model.
        metrics (dict): Dictionary with:
            - 'MSE': float, Mean Squared Error on test set
            - 'MAE': float, Mean Absolute Error on test set
            - 'R2': float, R² score on test set
        feature_data (pd.DataFrame): Feature matrix `X` used for model training.

    Doctest:
        >>> import pandas as pd
        >>> data = {
        ...     'LapNumber': [1, 2, 3, 4, 5],
        ...     'LapTime': [90.0, 91.5, 89.2, 92.1, 90.7],
        ...     'FreshTyre': [1, 0, 1, 0, 1],
        ...     'IsPersonalBest': [1, 0, 0, 1, 0],
        ...     'Throttle': [80, 82, 78, 85, 83],
        ...     'Brake': [0, 1, 1, 0, 1],
        ...     'DRS': [2, 3, 1, 4, 3],
        ...     'Compound': [5, 5, 6, 6, 5],
        ...     'TyreLife': [5, 6, 7, 8, 9],
        ...     'TrackStatus': [0, 1, 0, 0, 1],
        ...     'Weight': [820, 819, 818, 817, 816],
        ...     'PitDuration': [0.0, 0.0, 0.0, 20.3, 0.0],
        ...     'TrackTemp': [35.0, 34.5, 36.2, 33.9, 34.8]
        ... }
        >>> df_test = pd.DataFrame(data)
        >>> model, metrics, X = train_random_forest(df_test) # doctest: +ELLIPSIS
        Random Forest Regression Results:
        ...
        >>> X.shape[1] == 9  # 13 cols - 4 dropped: ['LapNumber', 'LapTime', 'FreshTyre', 'IsPersonalBest']
        True
        >>> all(isinstance(v, (int, float)) for v in metrics.values())
        True
    """
    # 1. Separate features and target
    X = df.drop(['LapNumber','LapTime', 'FreshTyre', 'IsPersonalBest'], axis=1)
    y = df['LapTime']

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2 #, random_state=1
    )

    # 3. Initialize and train model
    model = RandomForestRegressor(max_depth=40, n_estimators=300, min_samples_split=2, min_samples_leaf=1)
    model.fit(X_train, y_train)

    # 4. Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Random Forest Regression Results:")
    print(f"• Mean Squared Error (MSE): {mse:.2f}")
    print(f"• Mean Absolute Error (MAE): {mae:.2f}")
    print(f"• R² Score: {r2:.4f}")

    metrics = {
        'MSE': mse,
        'MAE': mae,
        'R2': r2
    }

    return model, metrics, X

def final_cleaning_for_model(
    input_path="data/regression_final.csv",
    output_path="data/regression_final_cleaned.csv"
):
    """
    Applies final preprocessing steps: filtering flags, encoding track status,
    filling missing values, and dropping unnecessary columns.
    """
    df = pd.read_csv(input_path)

    # 1. Drop red flag laps
    df = df[~df["TrackStatus"].astype(str).str.contains("5")].copy()

    # 2. Encode TrackStatus
    def map_status(status_str):
        status_str = str(status_str)
        if "4" in status_str:
            return 3  # Safety Car
        elif "6" in status_str:
            return 2  # Virtual Safety Car
        elif "2" in status_str:
            return 1  # Yellow Flag
        else:
            return 0  # All Clear

    df["TrackStatus"] = df["TrackStatus"].apply(map_status)

    # 3. Drop unreliable gear column (optional)
    if "nGear" in df.columns:
        df.drop("nGear", axis=1, inplace=True)

    # 4. Fill missing values
    df["PitDuration"] = df["PitDuration"].fillna(0)
    df["TrackTemp"] = df["TrackTemp"].fillna(df["TrackTemp"].median())
    df["TyreLife"] = df["TyreLife"].fillna(df["TyreLife"].median())
    df["Compound"] = df["Compound"].fillna(df["Compound"].mode()[0])

    # 5. Save cleaned data
    df.to_csv(output_path, index=False)
    print(f" Cleaned regression data saved to {output_path}")
    return df

def randomize_tracktemp(df_random,
                        fluctuation_std=0.3,
                        min_temp_clip=30,
                        max_temp_clip=45,
                        cooling_after_lap=None,
                        cooling_bias=-0.05):
    """
    Randomly generate realistic TrackTemp values for 78 laps.

    Parameters:
    - df_random: DataFrame containing at least 'LapNumber' and 'TrackTemp'.
    - fluctuation_std: Standard deviation of lap-to-lap temperature change (°C).
    - min_temp_clip: Minimum allowed TrackTemp (°C).
    - max_temp_clip: Maximum allowed TrackTemp (°C).
    - cooling_after_lap: If set, apply slight cooling drift after this lap.
    - cooling_bias: Mean shift for lap-to-lap fluctuation after cooling lap (°C).

    Returns:
    - tracktemp_randomized_df: DataFrame with LapNumber and randomized TrackTemp.
    """

    randomized_tracktemps = []

    # Step 1: Pick Lap 1
    lap1_data = df_random[df_random['LapNumber'] == 1]['TrackTemp']
    lap1_temp = lap1_data.sample(n=1, random_state=np.random.randint(10000)).values[0]
    randomized_tracktemps.append(lap1_temp)

    # Step 2: Generate laps 2–78
    for lap in range(2, 79):
        prev_temp = randomized_tracktemps[-1]

        if cooling_after_lap is not None and lap > cooling_after_lap:
            fluctuation = np.random.normal(loc=cooling_bias, scale=fluctuation_std)
        else:
            fluctuation = np.random.normal(loc=0, scale=fluctuation_std)

        new_temp = prev_temp + fluctuation
        new_temp = np.clip(new_temp, min_temp_clip, max_temp_clip)

        randomized_tracktemps.append(new_temp)

    # Assemble final DataFrame
    tracktemp_randomized_df = pd.DataFrame({
        'LapNumber': np.arange(1, 79),
        'TrackTemp': randomized_tracktemps
    })

    return tracktemp_randomized_df

def generate_weight_per_lap(car_type='V10'):
    """
    Generate a dataframe of Weight per LapNumber for a given car type.

    Args:
        car_type (str): Either 'V6' or 'V10', representing the type of F1 engine.

    Returns:
        pd.DataFrame: A dataframe with 78 rows and two columns:
            - 'LapNumber' (int): Lap number from 1 to 78
            - 'Weight' (float): Total car weight including fuel at each lap

    Raises:
        ValueError: If car_type is not 'V6' or 'V10'

    Doctest:
        >>> df_weight = generate_weight_per_lap('v10')
        >>> lap20 = df_weight[df_weight['LapNumber'] == 20]['Weight'].values[0]
        >>> lap70 = df_weight[df_weight['LapNumber'] == 70]['Weight'].values[0]
        >>> lap70 < lap20
        np.True_
    """

    # Define base weights for each car type
    base_weights = {
        'V6': 798,   # Base weight for V6 hybrid car (without fuel)
        'V10': 783   # Base weight for V10 hybrid car (without fuel)
    }

    # Normalize car_type input
    car_type = car_type.upper()

    # Validate input
    if car_type not in base_weights:
        raise ValueError("car_type must be 'V6' or 'V10'")

    base_weight = base_weights[car_type]

    # Constants
    laps_total = 78
    fuel_start_min = 100
    fuel_start_max = 110
    fuel_end_min = 1
    fuel_end_max = 2

    # Generate randomized fuel values
    initial_fuel = np.random.uniform(fuel_start_min, fuel_start_max)
    final_fuel = np.random.uniform(fuel_end_min, fuel_end_max)
    burn_per_lap = (initial_fuel - final_fuel) / (laps_total - 1)

    # Generate weight per lap
    weights = []
    for lap in range(1, laps_total + 1):
        current_fuel = initial_fuel - (lap - 1) * burn_per_lap
        current_weight = base_weight + current_fuel
        weights.append((lap, current_weight))

    # Return DataFrame
    weight_df = pd.DataFrame(weights, columns=['LapNumber', 'Weight'])
    return weight_df

def generate_trackstatus_per_lap():
    """
    Generate a DataFrame with randomized TrackStatus per LapNumber for 78 laps.

    Returns:
    - trackstatus_df: DataFrame with columns ['LapNumber', 'TrackStatus']
    """

    # Constants
    laps_total = 78
    trackstatus_choices = [0, 1, 2, 3]  # 0: AC, 1: YF, 2: VSC, 3: SC
    probabilities = [0.8957, 0.0762, 0.0064, 0.0217]  # From your EDA

    # Randomly choose one status per lap
    trackstatus_list = np.random.choice(trackstatus_choices, size=laps_total, p=probabilities)

    # Create dataframe
    trackstatus_df = pd.DataFrame({
        'LapNumber': np.arange(1, laps_total + 1),
        'TrackStatus': trackstatus_list
    })

    return trackstatus_df

def generate_throttle_rpm_per_lap(df_random, car_type='V10', transition_throttle=60):
    """
    Generate a dataframe with randomized Throttle and calculated RPM per lap
    for either V6 or V10 engine type.

    Args:
    - df_random: Input DataFrame with 'LapNumber' and 'Throttle' columns
    - car_type: 'V6' or 'V10'
    - transition_throttle: Throttle % at which model switches between quadratic and linear

    Returns:
    - result_df: DataFrame with ['LapNumber', 'Throttle', 'RPM']
    """

    # Normalize input
    car_type = car_type.upper()

    # Constants
    laps_total = 78

    # Coefficients from piecewise regression model
    # For Throttle ≤ transition_throttle (Quadratic)
    quad_intercept = 3497.86
    quad_coef1 = 219.56
    quad_coef2 = -2.03

    # For Throttle > transition_throttle (Linear)
    lin_intercept = 7224.98
    lin_coef = -72.24

    lap_results = []

    for lap in range(1, laps_total + 1):
        lap_data = df_random[df_random['LapNumber'] == lap]

        if lap_data.empty:
            continue

        sampled_throttle = lap_data['Throttle'].sample(n=1).values[0]

        # Predict RPM based on throttle
        if sampled_throttle <= transition_throttle:
            rpm = (quad_intercept
                   + quad_coef1 * sampled_throttle
                   + quad_coef2 * sampled_throttle**2)
        else:
            rpm = lin_intercept + lin_coef * sampled_throttle

        # Scale only if V10
        if car_type == 'V10':
            rpm *= (20000 / 15000)  # Scale up by 1.333...

        lap_results.append((lap, sampled_throttle, rpm))

    result_df = pd.DataFrame(lap_results, columns=['LapNumber', 'Throttle', 'RPM'])

    return result_df

def generate_full_lap_data(laps, df_random, car_type):
    """
    Generates a full dataset per lap by combining fixed variables (laps) with
    randomized TrackTemp, Weight, TrackStatus, Throttle, and RPM.

    Args:
    - laps: DataFrame containing fixed variables and 'LapNumber'.
    - df_random: DataFrame containing random pool for throttle (must have 'LapNumber' and 'Throttle').

    Returns:
    - final_df: DataFrame with all fixed and random variables combined, LapNumber dropped.
    """

    # 1. Generate random variables
    tracktemp_df = randomize_tracktemp(df_random)
    weight_df = generate_weight_per_lap(car_type)
    trackstatus_df = generate_trackstatus_per_lap()
    throttle_rpm_df = generate_throttle_rpm_per_lap(df_random,car_type)
    # 2. Merge everything on 'LapNumber'
    merged_df = laps.merge(tracktemp_df, on='LapNumber', how='left')
    merged_df = merged_df.merge(weight_df, on='LapNumber', how='left')
    merged_df = merged_df.merge(trackstatus_df, on='LapNumber', how='left')
    merged_df = merged_df.merge(throttle_rpm_df, on='LapNumber', how='left')

    # 3. Drop LapNumber
    merged_df = merged_df.drop(columns=['LapNumber'])

    return merged_df

def create_fixed_variables(df_source):
    laps = pd.DataFrame({'LapNumber': range(1, 79)})

    # ---- Step 2: Aggregate fixed variables from source ----
    agg_fixed = df_source.groupby('LapNumber').agg({
        'Throttle': 'mean',
        'Brake': 'mean',
        'DRS': 'mean'
    }).reset_index()

    # Round Throttle, Brake, DRS to integers
    agg_fixed['Brake'] = agg_fixed['Brake'].round().astype(int)
    agg_fixed['DRS'] = agg_fixed['DRS'].round().astype(int)

    # Merge into the lap DataFrame
    laps = laps.merge(agg_fixed, on='LapNumber', how='left')

    # ---- Step 3: Pit Duration ----
    # Average of nonzero pit_durations
    avg_pit_duration = df_source[df_source['PitDuration'] > 0]['PitDuration'].mean()

    # Assign pit_duration
    laps['PitDuration'] = 0.0
    laps.loc[laps['LapNumber'] == 32, 'PitDuration'] = avg_pit_duration

    # ---- Step 4: Compound ----
    laps['Compound'] = np.where(laps['LapNumber'] <= 31, 5, 6)

    # ---- Step 5: Tyre Life ----
    laps['TyreLife'] = np.where(
        laps['LapNumber'] <= 31,
        laps['LapNumber'],                   # Laps 1–31
        laps['LapNumber'] - 31                # Restart count from 1 after pit stop
    )
    laps.drop(columns=['Throttle'], inplace=True)
    return laps

def monte_carlo_simulation(model, laps, df_random, car_type, n_simulations=5000):
    """
    Run Monte Carlo simulations to predict race times and output a DataFrame.

    Args:
    - model: Trained Random Forest model
    - laps: DataFrame with fixed variables (must include 'LapNumber')
    - df_random: DataFrame with random pool for throttle and tracktemp generation
    - n_simulations: Number of Monte Carlo simulations to run (default=5000)

    Returns:
    - race_times_df: DataFrame with columns ['SimulationNumber', 'RaceTime']
    """

    simulation_numbers = []
    race_times = []

    for sim in range(1, n_simulations + 1):
        # Step 1: Generate a new randomized lap dataset
        sim_df = generate_full_lap_data(laps.copy(), df_random, car_type)

        correct_feature_order = [
        'TrackTemp', 'RPM', 'Throttle', 'Brake', 'DRS',
        'Compound', 'TyreLife', 'TrackStatus', 'Weight', 'PitDuration'
        ]
        sim_df = sim_df[correct_feature_order]
        # Step 2: Predict lap times
        lap_times = model.predict(sim_df)

        # Step 3: Sum up all 78 lap times to get total race time
        total_race_time = np.sum(lap_times)

        # Step 4: Store the simulation number and race time
        simulation_numbers.append(sim)
        race_times.append(total_race_time)

    # Create DataFrame
    race_times_df = pd.DataFrame({
        'SimulationNumber': simulation_numbers,
        'RaceTime': race_times
    })

    return race_times_df

def t_test_hypothesis1(df_v10, df_v6):
    # Extract race times
    v10_times = df_v10['RaceTime']
    v6_times = df_v6['RaceTime']

    # Perform t-test (one-tailed: testing if V10 is faster than V6)
    t_stat, p_value = ttest_ind(v10_times, v6_times, alternative='less')  # H1: V10 < V6

    # Output results
    print("--- T-Test Results ---")
    print(f"T-statistic = {t_stat:.4f}")
    print(f"P-value = {p_value:.10f}")

    if p_value < 0.05:
        print(" **Reject Null Hypothesis**: V10 hybrid engines yield significantly faster race times than V6 (p < 0.05).")
    else:
        print("**Fail to Reject Null Hypothesis**: No significant evidence that V10 is faster than V6 (p ≥ 0.05).")

    # Calculate means
    mean_v10 = v10_times.mean()
    mean_v6 = v6_times.mean()
    print(f"\nMean Race Time (V10): {mean_v10:.2f} seconds")
    print(f"Mean Race Time (V6): {mean_v6:.2f} seconds")
    print(f"Difference: {mean_v6 - mean_v10:.2f} seconds (V6 - V10)")

def simulate_lapwise_weight_threshold_effect(model, laps, df_random, car_type='V10', n_samples=5000, weight_threshold=700):
    """
    Monte Carlo test of per-lap weight threshold effect using lap-wise comparison.

    Args:
    - model: trained regression model
    - laps: fixed lap template
    - df_random: random input source
    - car_type: 'V6' or 'V10'
    - n_samples: number of simulations
    - weight_threshold: threshold in kg for separating groups

    Returns:
    - low_weight_laps: all lap times with weight < threshold
    - high_weight_laps: all lap times with weight ≥ threshold
    """
    low_weight_laps = []
    high_weight_laps = []

    for _ in range(n_samples):
        sim_df = generate_full_lap_data(laps.copy(), df_random, car_type=car_type)

        # Ensure correct column order
        feature_cols = ['TrackTemp', 'RPM', 'Throttle', 'Brake', 'DRS',
                        'Compound', 'TyreLife', 'TrackStatus', 'Weight', 'PitDuration']
        sim_df = sim_df[feature_cols]

        # Predict lap times
        lap_times = model.predict(sim_df)

        # Append lap times to appropriate group
        for lap_time, weight in zip(lap_times, sim_df['Weight']):
            if weight < weight_threshold:
                low_weight_laps.append(lap_time)
            else:
                high_weight_laps.append(lap_time)

    # Perform one-tailed t-test
    t_stat, p_value = ttest_ind(low_weight_laps, high_weight_laps, alternative='less')

    print(f"\n--- Per-Lap Weight Threshold Test at {weight_threshold}kg ---")
    print(f"Low Weight Lap Count:  {len(low_weight_laps)}")
    print(f"High Weight Lap Count: {len(high_weight_laps)}")
    print(f"Mean Lap Time (< {weight_threshold}kg):  {np.mean(low_weight_laps):.4f} s")
    print(f"Mean Lap Time (≥ {weight_threshold}kg): {np.mean(high_weight_laps):.4f} s")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.6f}")

    if p_value < 0.05:
        print(" Reject Null Hypothesis: Lower weight leads to significantly faster laps.")
    else:
        print("Fail to Reject Null: No significant lap time difference based on weight.")

    return low_weight_laps, high_weight_laps
