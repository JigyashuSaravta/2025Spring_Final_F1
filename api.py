import requests
import pandas as pd

years = [2021, 2022, 2023]
data_types = ["pit", "race_control"]
country = "Monaco"

def get_meeting_key(year):
    url = f"https://api.openf1.org/v1/meetings?year={year}"
    response = requests.get(url)
    if response.status_code == 200 and response.json():
        for meeting in response.json():
            if meeting.get("country_name") == country:
                return meeting["meeting_key"]
        print(f"❌ Monaco not found in meetings for {year}")
        return None
    else:
        print(f"❌ Could not fetch meetings for {year}")
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

# MAIN LOOP
for year in years:
    meeting_key = get_meeting_key(year)
    if not meeting_key:
        continue

    session_key = get_session_key(meeting_key, session_name="Race")
    if not session_key:
        continue

    for data_type in data_types:
        download_and_save(data_type, session_key, year)
