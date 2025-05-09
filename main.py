import os
import pandas as pd
import fastf1
import re
from glob import glob


def fetch_lap_data(years, save_dir="."):
    """
    Fetches lap data for Monaco GP for the given years and saves them as CSVs.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for year in years:
        try:
            print(f"Fetching lap data for Monaco {year}...")
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
    Fetches car telemetry data per driver using public FastF1 API.
    Output is saved in the same format as legacy car_data_YYYY_driver_XX.csv files.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for year in years:
        print(f"Fetching car data for Monaco {year}...")
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
    match = re.search(r'driver_(\d+)', filename)
    return match.group(1) if match else None

def annotate_lap_number(year, data_dir="."):
    print(f"Annotating lap numbers for {year}...")

    lap_file = os.path.join(data_dir, f"monaco_laps_{year}.csv")
    df_lap = pd.read_csv(lap_file)
    df_lap['Time'] = pd.to_timedelta(df_lap['Time'])

    # First lap time to use as timeline anchor
    lap_start_time = df_lap['Time'].min()

    car_files = sorted(glob(os.path.join(data_dir, f"car_data_{year}_driver_*.csv")))
    car_data_frames = []

    for car_file in car_files:
        driver_number = extract_driver_number(car_file)
        df_car = pd.read_csv(car_file)

        # Align telemetry clock to official lap time clock
        car_start_date = pd.to_datetime(df_car['Date'].iloc[0])
        df_car['Time'] = pd.to_datetime(df_car['Date']) - car_start_date + lap_start_time

        df_car['DriverNumber'] = int(driver_number)
        car_data_frames.append(df_car)

    if not car_data_frames:
        print(f"No car data found for {year}. Skipping...")
        return

    df_car = pd.concat(car_data_frames, ignore_index=True)
    df_car['LapNumber'] = None

    df_lap = df_lap[['DriverNumber', 'LapNumber', 'Time']].copy().sort_values(['DriverNumber', 'LapNumber'])

    for driver in df_lap['DriverNumber'].unique():
        driver_laps = df_lap[df_lap['DriverNumber'] == driver].sort_values('LapNumber')
        driver_car_data = df_car['DriverNumber'] == driver

        for i in range(len(driver_laps) - 1):
            lap_start = driver_laps.iloc[i]['Time']
            lap_end = driver_laps.iloc[i + 1]['Time']
            lap_number = driver_laps.iloc[i]['LapNumber']

            in_lap = driver_car_data & (df_car['Time'] >= lap_start) & (df_car['Time'] < lap_end)
            df_car.loc[in_lap, 'LapNumber'] = lap_number

        # Assign final lap
        last_lap = driver_laps.iloc[-1]
        in_last_lap = driver_car_data & (df_car['Time'] >= last_lap['Time'])
        df_car.loc[in_last_lap, 'LapNumber'] = last_lap['LapNumber']

    df_car = df_car.dropna(subset=['LapNumber'])

    output_file = os.path.join(data_dir, f"car_data_with_lap_{year}.csv")
    df_car.to_csv(output_file, index=False)
    print(f"Saved annotated car data to {output_file} with {len(df_car)} rows.")

def aggregate_car_data(years, data_dir=".", output_prefix="aggregated_car_data"):
    all_dfs = []

    for year in years:
        print(f"Aggregating car data for {year}...")
        file_path = os.path.join(data_dir, f"car_data_with_lap_{year}.csv")
        try:
            df = pd.read_csv(file_path)

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

        except FileNotFoundError:
            print(f"File not found for {year}. Skipping...")

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
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for year in years:
        try:
            print(f"Fetching weather data for Monaco {year}...")
            session = fastf1.get_session(year, 'Monaco', 'Race')
            session.load(telemetry=False, weather=True)

            df_weather = session.weather_data.copy()
            df_weather.to_csv(os.path.join(save_dir, f"weather_{year}.csv"), index=False)
            print(f"  Saved weather_{year}.csv")
        except Exception as e:
            print(f"❌ Error fetching weather for {year}: {e}")

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
            print(f"❌ Could not process weather for {year}: {e}")

    df_all = pd.concat(aggregated_weather_dfs, ignore_index=True)
    df_all.to_csv(output_path, index=False)
    print(f"✅ Aggregated weather saved to {output_path}")
    return df_all

def concat_lap_metadata(years, data_dir=".", output_path="lap_metadata_all_years.csv"):
    """
    Concatenates lap metadata files for all years into a single dataframe and saves to CSV.
    """
    lap_dfs = []
    for year in years:
        file_path = os.path.join(data_dir, f"monaco_laps_{year}.csv")
        df = pd.read_csv(file_path)
        df["Year"] = year
        lap_dfs.append(df)

    df_laps = pd.concat(lap_dfs, ignore_index=True)
    df_laps.to_csv(os.path.join(data_dir, output_path), index=False)
    print(f"✅ Saved all-year lap metadata to {output_path}")
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
    print(f"✅ Merged car + lap data saved to {output_path}")
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
    print(f"✅ Final full dataset saved to {output_path}")
    return final_df

import os

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

    print(f"✅ Cleanup complete. Kept only: {keep_filename}")
    if deleted_files:
        print(f"🗑️ Deleted {len(deleted_files)} files:")
        for f in deleted_files:
            print(f"  - {f}")
