import fastf1
from fastf1.events import get_event_schedule
from fastf1.api import car_data, track_status_data, lap_count, driver_info, weather_data
import pandas as pd
import os

# Save everything directly into the same folder as this script
SAVE_DIR = os.path.dirname(__file__)

years = range(2018, 2025)  # 2020 will be skipped gracefully

for year in years:
    try:
        print(f"🔄 Loading Monaco {year}")
        schedule = get_event_schedule(year)
        monaco = schedule[schedule['EventName'].str.contains("Monaco", case=False)]

        if monaco.empty:
            print(f"❌ No Monaco GP found for {year}")
            continue

        round_num = int(monaco.iloc[0]['RoundNumber'])
        session = fastf1.get_session(year, round_num, 'R')
        session.load()
        path = session.api_path

        # ---- Car Data ----
        car_dict = car_data(path)
        for driver_number, df in car_dict.items():
            df.to_csv(os.path.join(SAVE_DIR, f"car_data_{year}_driver_{driver_number}.csv"), index=False)

        # ---- Track Status ----
        track = pd.DataFrame(track_status_data(path))
        track.to_csv(os.path.join(SAVE_DIR, f"track_status_{year}.csv"), index=False)

        # ---- Lap Count ----
        lap = pd.DataFrame(lap_count(path))
        lap.to_csv(os.path.join(SAVE_DIR, f"lap_count_{year}.csv"), index=False)

        # ---- Driver Info ----
        driver_df = pd.DataFrame.from_dict(driver_info(path), orient='index')
        driver_df.to_csv(os.path.join(SAVE_DIR, f"driver_info_{year}.csv"))

        # ---- Weather ----
        weather_df = pd.DataFrame(weather_data(path))
        weather_df.to_csv(os.path.join(SAVE_DIR, f"weather_{year}.csv"), index=False)

        print(f"✅ Done Monaco {year}")

    except Exception as e:
        print(f"⚠️ Failed {year}: {e}")
