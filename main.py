import requests
import pandas as pd
import numpy as np
import regex as re

years = [2021, 2022, 2023, 2024]
data_types = ["laps", "stints"]
country = "Monaco"

def get_meeting_key(year):
    url = f"https://api.openf1.org/v1/meetings?year={year}&country_name={country}"
    response = requests.get(url)
    if response.status_code == 200 and response.json():
        return response.json()[0]['meeting_key']
    else:
        print(f"❌ Could not fetch meeting_key for {year}")
        return None

def get_session_key(meeting_key, session_name="Race"):
    url = f"https://api.openf1.org/v1/sessions?meeting_key={meeting_key}&session_name={session_name}"
    response = requests.get(url)
    if response.status_code == 200 and response.json():
        return response.json()[0]['session_key']
    else:
        print(f"❌ Could not fetch session_key for meeting_key {meeting_key}")
        return None

def download_and_save(endpoint, session_key, year):
    url = f"https://api.openf1.org/v1/{endpoint}?session_key={session_key}"
    print(f"📡 Fetching {endpoint} for Monaco {year} (session_key={session_key})...")
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data:
            df = pd.DataFrame(data)
            filename = f"monaco_{year}_{endpoint}.csv"
            df.to_csv(filename, index=False)
            print(f"✅ Saved {filename}")
        else:
            print(f"⚠️ No data for {endpoint} in {year}")
    else:
        print(f"❌ Failed to fetch {endpoint} for {year}: HTTP {response.status_code}")

def fetch_driver_csv():
    df_list = []

    for num in range(1, 82):
        try:
            file_name = f"monaco_2023_car_data_driver_{num}.csv"
            df = pd.read_csv(file_name)
            df['driver_number'] = num  # add the driver number explicitly
            df_list.append(df)
            print(f"Loaded: {file_name}")
        except FileNotFoundError:
            print(f"Skipping: {file_name} (not found)")
        except Exception as e:
            print(f"Error loading {file_name}: {e}")

    # inside fetch_driver_csv()
    if df_list:
        return pd.concat(df_list, ignore_index=True)  # <- no comma here!
    else:
        return pd.DataFrame()

def fix_if_not_iso8601(s):
    iso8601_regex = re.compile(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+\d{2}:\d{2}$"
    )
    if isinstance(s, str) and not iso8601_regex.fullmatch(s):
        # If it's missing microseconds and has timezone, patch it
        return s.replace('+00:00', '.000000+00:00')
    else:
        return s

def clean_data(df_drive, df_lap):

    df_drive['date'] = df_drive['date'].astype(str)
    df_lap['date_start'] = df_lap['date_start'].astype(str)

    df_drive['date'] = df_drive['date'].apply(fix_if_not_iso8601)
    df_lap['date_start'] = df_lap['date_start'].apply(fix_if_not_iso8601)

    # Create an empty lap_number column in df_drive
    df_drive['lap_number'] = np.nan

    df_drive['date'] = pd.to_datetime(df_drive['date'], utc=True)
    df_lap['date_start'] = pd.to_datetime(df_lap['date_start'], utc=True)

    RACE_START = pd.Timestamp('2023-05-28T13:03:13.519001+00:00')
    sector_cols = ['duration_sector_1', 'duration_sector_2', 'duration_sector_3']

    # Iterate group-wise for matching meeting/session/driver
    for (m_key, s_key, d_num), group in df_lap.groupby(['meeting_key', 'session_key', 'driver_number']):
        # Inside the group loop
        group = group.sort_values('lap_number').reset_index()  # keep original index for writing back
        # original_indices = group['index']  # these are df_lap indices

        if pd.isna(group.loc[0, 'date_start']):
            # lap1_idx = original_indices[0]
            # df_lap.at[lap1_idx, 'date_start'] = RACE_START
            group.loc[0, 'date_start'] = RACE_START  # update local copy too

        for i in range(len(group)):
            lap_num = group.loc[i, 'lap_number']
            # print(lap_num)
            start_time = group.loc[i, 'date_start']

            # Compute end_time
            if i < len(group) - 1:
                end_time = group.iloc[i + 1]['date_start']
            else:
                # Last lap: compute total duration in seconds and convert to timedelta
                sector_times = group.loc[i, sector_cols]
                if sector_times.notna().all():
                    total_seconds = sector_times.sum()
                    end_time = start_time + pd.to_timedelta(total_seconds, unit='s')
                else:
                    # Fallback to max drive time
                    end_time = df_drive[
                        (df_drive['meeting_key'] == m_key) &
                        (df_drive['session_key'] == s_key) &
                        (df_drive['driver_number'] == d_num)
                        ]['date'].max()

            # Apply annotation to df_drive
            mask = (
                    (df_drive['meeting_key'] == m_key) &
                    (df_drive['session_key'] == s_key) &
                    (df_drive['driver_number'] == d_num) &
                    (df_drive['date'] >= start_time) &
                    (df_drive['date'] < end_time)
            )
            df_drive.loc[mask, 'lap_number'] = lap_num


    # Finalize column type
    df_drive['lap_number'] = df_drive['lap_number'].astype('Int64')
    df_drive.dropna(subset=['lap_number'], inplace=True)
    return df_drive, df_lap

def aggregate_df(df_drive):
    df_drive['brake_count'] = (df_drive['brake'] == 100).astype(int)

    # DRS usage count (only when it's 10, 12, or 14)
    df_drive['drs_count'] = df_drive['drs'].isin([10, 12, 14]).astype(int)

    # Group and aggregate
    agg_df = df_drive.groupby(['driver_number', 'lap_number']).agg({
        'brake_count': 'sum',
        'n_gear': 'mean',
        'drs_count': 'sum',
        'speed': 'mean',
        'throttle': 'mean',
        'rpm': 'mean'
    }).reset_index()
    return agg_df

def merge_data(df_drive, df_lap):
    # Perform right join
    merged_df = pd.merge(
        df_drive,
        df_lap,
        on=['driver_number', 'lap_number'],
        how='right'
    )

    # Optional: sort by driver and lap
    merged_df = merged_df.sort_values(by=['driver_number', 'lap_number'])

    return merged_df

if __name__ == "__main__":
    df_drive = fetch_driver_csv()
    df_lap = pd.read_csv('monaco_2023_laps.csv')
    df_drive_clean, df_lap_clean = clean_data(df_drive, df_lap)
    agg_df = aggregate_df(df_drive_clean)
    merged_df = merge_data(agg_df, df_lap_clean)
    df_yellow_flags = pd.read_csv('yellow_flags.csv')
    merged_df_1 = pd.merge(merged_df, df_yellow_flags, on=['lap_number'], how='left')

    print(df_drive_clean.shape)
    print(df_lap_clean.shape)
    print(merged_df.shape)
    print(merged_df_1.shape)

    merged_df_1.to_csv('df_regression.csv', index=False)