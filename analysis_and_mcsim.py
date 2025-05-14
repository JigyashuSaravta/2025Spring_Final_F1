from main import *
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

YEARS = [2018, 2019, 2021, 2022, 2023, 2024]
SAVE_DIR = "data"

fetch_lap_data(YEARS, save_dir=SAVE_DIR)
fetch_car_data(YEARS, save_dir=SAVE_DIR)

fetch_weather_data(YEARS, save_dir="data")
df_weather_all_years = annotate_and_aggregate_weather(YEARS, lap_data_dir="data", weather_dir="data")

df_aggregate_car = aggregate_car_data(years=YEARS, data_dir="data")

# Step 1
df_laps_all = concat_lap_metadata(YEARS, data_dir="data")

# Step 2
df_car_lap = merge_car_with_laps("data/aggregated_car_data_all_years.csv", df_laps_all, output_path="data/car_lap_merged.csv")

# Step 3
final_df = merge_with_weather("data/aggregated_weather_all_years.csv", df_car_lap, output_path="data/merged_lap_car_weather_all_years.csv")

cleanup_intermediate_csvs(directory="data")

df_weighted = add_simulated_weight_column(
    input_path="data/merged_lap_car_weather_all_years.csv",
    output_path="data/regression_ready_with_weight.csv"
)

final_df = preprocess_for_regression()

df_cleaned = final_cleaning_for_model(input_path='data/regression_final.csv')

model_normal, metrics, X = train_random_forest(df_cleaned)

# Try standard train/test to simulate CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = RandomForestRegressor(max_depth=40, n_estimators=300, min_samples_split=2, min_samples_leaf=1)

# 1. Separate features and target
X = df_cleaned.drop(['LapNumber','LapTime', 'FreshTyre', 'IsPersonalBest'], axis=1)
y = df_cleaned['LapTime']

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2 #, random_state=1
)

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    r2 = model.score(X_test, y_test)
    print(f"Fold {fold + 1} R²: {r2:.4f}")



def plot_feature_importances(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.show()

plot_feature_importances(model, X.columns)

df_fixed = create_fixed_variables(df_cleaned)

df_random = df_cleaned[['LapNumber','TrackTemp','Weight','TrackStatus','Throttle','RPM']]

race_times_df_v10 = monte_carlo_simulation(model, df_fixed, df_random, n_simulations=5000, car_type='v10')
race_times_df_v6 = monte_carlo_simulation(model, df_fixed, df_random, n_simulations=5000, car_type='v6')
# race_times_df_v6.to_csv('data/race_times_v6.csv')
# race_times_df_v10.to_csv('data/race_times_v10.csv')
#
# race_times_df_v6 = pd.read_csv('data/race_times_v6.csv')
# race_times_df_v10 = pd.read_csv('data/race_times_v10.csv')

t_test_hypothesis1(race_times_df_v10, race_times_df_v6)

low_laps, high_laps = simulate_lapwise_weight_threshold_effect(
    model=model,
    laps=df_fixed,
    df_random=df_random,
    car_type='V6',
    weight_threshold=800,
    n_samples=5000
)