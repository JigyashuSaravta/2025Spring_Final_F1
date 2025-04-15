import requests
import pandas as pd

years = [2021, 2022, 2023]
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

# Loop through years and fetch/save lap and stint data
# for year in years:
#     print(f"\n🔎 Year: {year}")
#     meeting_key = get_meeting_key(year)
#     if meeting_key:
#         session_key = get_session_key(meeting_key)
#         if session_key:
#             for data_type in data_types:
#                 download_and_save(data_type, session_key, year)
def fetch_driver_csv(num):
    file_name = f"monaco_2023_car_data_driver_{num}.csv"
    df = pd.read_csv(file_name)
    return df

def c

if __name__ == "__main__":
    df1 = fetch_driver_csv(1)
    print(df1.head())