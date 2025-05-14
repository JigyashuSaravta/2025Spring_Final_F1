# F1 Lap Time Prediction & Monte Carlo Simulation

This project uses real-world telemetry and weather data from the Formula 1 Monaco Grand Prix (2018–2024) to model race performance using a Random Forest Regressor and simulate multiple race outcomes using a Monte Carlo approach. It is built with `FastF1`, `scikit-learn`, and `pandas`.

## Project Structure

* `main.py`
  Contains **all core functions** for:

  * Fetching raw telemetry, lap, and weather data
  * Aggregating and merging datasets
  * Simulating realistic variables like car weight and track temperature
  * Preprocessing the dataset for machine learning
  * Training a Random Forest model
  * Running Monte Carlo simulations

* `analysis_and_mcsim.py`
  This is the **driver script**. It calls and executes the functions from `main.py` to:

  * Build the dataset
  * Train the model
  * Run simulations
  * Generate outputs and statistics

---

## Setup Instructions

### 1. Clone the Repository

git clone https://github.com/JigyashuSaravta/2025Spring_Final_F1.git

cd f1-laptime-mcsim

### 2. Install Required Packages

This project requires Python 3.9+ and the following libraries:

pip install fastf1 pandas numpy scikit-learn matplotlib

---

## Running the Project

This will fetch all data, preprocess it, and create the final merged dataset.

python analysis_and_mcsim.py


This script will:

* Download and save raw telemetry, lap, and weather data
* Generate features like weight, pit duration, and track temperature
* Preprocess the dataset for regression
* Train a Random Forest model
* Run 5000 simulations each for V6 and V10 hybrid car types
* Print results and visualizations

---

## Output

* Console output showing:
  * Regression model performance (MSE, MAE, R²)
  * Monte Carlo simulation results

---
