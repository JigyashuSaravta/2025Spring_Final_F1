# import requests
# import pandas as pd
#
# years = [2021, 2022, 2023]
# data_types = ["laps", "stints"]
# country = "Monaco"
#
# def get_meeting_key(year):
#     url = f"https://api.openf1.org/v1/meetings?year={year}&country_name={country}"
#     response = requests.get(url)
#     if response.status_code == 200 and response.json():
#         return response.json()[0]['meeting_key']
#     else:
#         print(f"❌ Could not fetch meeting_key for {year}")
#         return None
#
# def get_session_key(meeting_key, session_match="Race"):
#     url = f"https://api.openf1.org/v1/sessions?meeting_key={meeting_key}"
#     response = requests.get(url)
#     if response.status_code == 200:
#         sessions = response.json()
#         for s in sessions:
#             if s["session_name"].lower() == session_match.lower():
#                 return s["session_key"]
#         print(f"⚠️ No session matched '{session_match}' for meeting_key={meeting_key}")
#     else:
#         print(f"❌ Error fetching sessions for meeting_key={meeting_key}")
#     return None
#
# def download_and_save(endpoint, session_key, year):
#     url = f"https://api.openf1.org/v1/{endpoint}?session_key={session_key}"
#     print(f"📡 Fetching {endpoint} for Monaco {year} (session_key={session_key})...")
#     response = requests.get(url)
#     if response.status_code == 200:
#         data = response.json()
#         if data:
#             df = pd.DataFrame(data)
#             filename = f"monaco_{year}_{endpoint}.csv"
#             df.to_csv(filename, index=False)
#             print(f"✅ Saved {filename}")
#         else:
#             print(f"⚠️ No data for {endpoint} in {year}")
#     else:
#         print(f"❌ Failed to fetch {endpoint} for {year}: HTTP {response.status_code}")
#
# def get_driver_numbers(session_key):
#     url = f"https://api.openf1.org/v1/drivers?session_key={session_key}"
#     response = requests.get(url)
#     if response.status_code == 200:
#         data = response.json()
#         driver_numbers = sorted(set(d['driver_number'] for d in data))
#         print(f"✅ Found {len(driver_numbers)} drivers: {driver_numbers}")
#         return driver_numbers
#     else:
#         print("❌ Failed to fetch driver list.")
#         return []
#
# def download_car_data(session_key, driver_number, year):
#     url = f"https://api.openf1.org/v1/car_data?session_key={session_key}&driver_number={driver_number}"
#     print(f"📡 Fetching car_data for driver {driver_number}...")
#     response = requests.get(url)
#     if response.status_code == 200:
#         data = response.json()
#         if data:
#             df = pd.DataFrame(data)
#             filename = f"monaco_{year}_car_data_driver_{driver_number}.csv"
#             df.to_csv(filename, index=False)
#             print(f"✅ Saved {filename}")
#         else:
#             print(f"⚠️ No car_data for driver {driver_number}")
#     else:
#         print(f"❌ Failed to fetch car_data for driver {driver_number}: HTTP {response.status_code}")
#
# # Main loop
# for year in years:
#     print(f"\n🔎 Year: {year}")
#     meeting_key = get_meeting_key(year)
#     if meeting_key:
#         session_key = get_session_key(meeting_key)
#         if session_key:
#             for data_type in data_types:
#                 download_and_save(data_type, session_key, year)
#
#             # Special case: Get car data for Monaco 2023 only
#             if year == 2023:
#                 driver_numbers = get_driver_numbers(session_key)
#                 for driver_number in driver_numbers:
#                     download_car_data(session_key, driver_number, year)
