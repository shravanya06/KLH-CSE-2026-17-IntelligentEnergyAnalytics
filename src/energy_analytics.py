"""
Intelligent Energy Consumption Analytics for Smart Buildings
-------------------------------------------------------------
Clean source-code version of the project notebook.

Implemented components:
1. UCI Appliances Energy Prediction dataset loading
2. Data inspection and preprocessing
3. Time-based feature engineering
4. Exploratory energy-consumption analysis
5. IQR-based anomaly detection
6. Random Forest baseline model
7. Time-series lag features and chronological Random Forest model
8. Gradient Boosting model
9. Model evaluation and comparison
10. Feature importance
11. Energy-saving recommendations
12. CSV result generation
"""

# Install separately if needed in a notebook:
# !pip install -q ucimlrepo

from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

dataset = fetch_ucirepo(id=374)

X = dataset.data.features
y = dataset.data.targets

df = pd.concat([X, y], axis=1)

print("Dataset shape:", df.shape)


# ============================================================
# 2. DATA INSPECTION
# ============================================================

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistical summary:")
print(df.describe())


# ============================================================
# 3. DATE/TIME PREPROCESSING AND FEATURE ENGINEERING
# ============================================================

# The dataset stores date and time without a separating space
# in some versions, e.g. 2016-01-1117:00:00.
df["date"] = (
    df["date"]
    .astype(str)
    .str.replace(
        r"(\d{4}-\d{2}-\d{2})(\d{2}:)",
        r"\1 \2",
        regex=True
    )
)

df["date"] = pd.to_datetime(
    df["date"],
    format="%Y-%m-%d %H:%M:%S"
)

df["hour"] = df["date"].dt.hour
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

print("\nNew time-based features created successfully!")
print(df[["date", "hour", "day_of_week", "month", "is_weekend"]].head())


# ============================================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================================

hourly_energy = df.groupby("hour")["Appliances"].mean()

plt.figure(figsize=(10, 5))
plt.plot(hourly_energy.index, hourly_energy.values, marker="o")
plt.xlabel("Hour of the Day")
plt.ylabel("Average Energy Consumption (Wh)")
plt.title("Average Energy Consumption by Hour")
plt.xticks(range(24))
plt.grid(True)
plt.tight_layout()
plt.show()


weekend_analysis = df.groupby("is_weekend")["Appliances"].mean()

print("\nAverage energy consumption:")
print("Weekdays :", round(weekend_analysis[0], 2), "Wh")
print("Weekends :", round(weekend_analysis[1], 2), "Wh")

plt.figure(figsize=(8, 5))
plt.bar(
    ["Weekday", "Weekend"],
    [weekend_analysis[0], weekend_analysis[1]]
)
plt.xlabel("Day Type")
plt.ylabel("Average Energy Consumption (Wh)")
plt.title("Weekday vs Weekend Energy Consumption")
plt.tight_layout()
plt.show()


# ============================================================
# 5. IQR-BASED ANOMALY DETECTION
# ============================================================

Q1 = df["Appliances"].quantile(0.25)
Q3 = df["Appliances"].quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

df["is_anomaly"] = (
    (df["Appliances"] < lower_limit) |
    (df["Appliances"] > upper_limit)
)

anomaly_count = int(df["is_anomaly"].sum())

print("\nAnomaly Detection")
print("-----------------")
print("Lower limit:", round(lower_limit, 2), "Wh")
print("Upper limit:", round(upper_limit, 2), "Wh")
print("Number of anomalies:", anomaly_count)
print(
    "Percentage of anomalies:",
    round(anomaly_count / len(df) * 100, 2),
    "%"
)


# ============================================================
# 6. ANOMALY VISUALIZATION
# ============================================================

normal_data = df[~df["is_anomaly"]]
anomaly_data = df[df["is_anomaly"]]

plt.figure(figsize=(14, 5))

plt.scatter(
    normal_data["date"],
    normal_data["Appliances"],
    s=5,
    label="Normal"
)

plt.scatter(
    anomaly_data["date"],
    anomaly_data["Appliances"],
    s=10,
    label="Anomaly"
)

plt.axhline(
    y=upper_limit,
    linestyle="--",
    label="Anomaly Threshold"
)

plt.xlabel("Date")
plt.ylabel("Energy Consumption (Wh)")
plt.title("Energy Consumption Anomaly Detection")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ============================================================
# 7. BASELINE RANDOM FOREST MODEL
# ============================================================

features = [
    "lights",
    "T1", "RH_1",
    "T2", "RH_2",
    "T3", "RH_3",
    "T4", "RH_4",
    "T5", "RH_5",
    "T6", "RH_6",
    "T7", "RH_7",
    "T8", "RH_8",
    "T9", "RH_9",
    "T_out",
    "Press_mm_hg",
    "RH_out",
    "Windspeed",
    "Visibility",
    "Tdewpoint",
    "hour",
    "day_of_week",
    "month",
    "is_weekend"
]

X = df[features]
y = df["Appliances"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest model...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nBaseline Random Forest Evaluation")
print("---------------------------------")
print("MAE :", round(mae, 2), "Wh")
print("RMSE:", round(rmse, 2), "Wh")
print("R² Score:", round(r2, 4))


# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nTop 10 factors influencing energy consumption:")
print(importance.head(10))

top_features = importance.head(10).sort_values()

plt.figure(figsize=(10, 6))
plt.barh(top_features.index, top_features.values)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 10 Factors Influencing Energy Consumption")
plt.tight_layout()
plt.show()


# ============================================================
# 9. TIME-SERIES LAG FEATURES
# ============================================================

df["lag_1"] = df["Appliances"].shift(1)
df["lag_6"] = df["Appliances"].shift(6)
df["rolling_mean_6"] = (
    df["Appliances"]
    .shift(1)
    .rolling(window=6)
    .mean()
)

df_ml = df.dropna().copy()

improved_features = features + [
    "lag_1",
    "lag_6",
    "rolling_mean_6"
]

X_time = df_ml[improved_features]
y_time = df_ml["Appliances"]

split_index = int(len(df_ml) * 0.8)

X_train_time = X_time.iloc[:split_index]
X_test_time = X_time.iloc[split_index:]

y_train_time = y_time.iloc[:split_index]
y_test_time = y_time.iloc[split_index:]


# ============================================================
# 10. TIME-SERIES RANDOM FOREST
# ============================================================

improved_model = RandomForestRegressor(
    n_estimators=150,
    random_state=42,
    n_jobs=-1
)

print("\nTraining improved time-series Random Forest...")
improved_model.fit(X_train_time, y_train_time)

y_pred_time = improved_model.predict(X_test_time)

mae_time = mean_absolute_error(y_test_time, y_pred_time)
rmse_time = np.sqrt(mean_squared_error(y_test_time, y_pred_time))
r2_time = r2_score(y_test_time, y_pred_time)

print("\nImproved Time-Series Random Forest Evaluation")
print("----------------------------------------------")
print("MAE :", round(mae_time, 2), "Wh")
print("RMSE:", round(rmse_time, 2), "Wh")
print("R² Score:", round(r2_time, 4))


# ============================================================
# 11. GRADIENT BOOSTING MODEL
# ============================================================

gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

print("\nTraining Gradient Boosting model...")
gb_model.fit(X_train_time, y_train_time)

y_pred_gb = gb_model.predict(X_test_time)

mae_gb = mean_absolute_error(y_test_time, y_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test_time, y_pred_gb))
r2_gb = r2_score(y_test_time, y_pred_gb)

print("\nGradient Boosting Evaluation")
print("----------------------------")
print("MAE :", round(mae_gb, 2), "Wh")
print("RMSE:", round(rmse_gb, 2), "Wh")
print("R² Score:", round(r2_gb, 4))


# ============================================================
# 12. NAIVE PREVIOUS-VALUE BASELINE
# ============================================================

naive_predictions = X_test_time["lag_1"]

naive_mae = mean_absolute_error(y_test_time, naive_predictions)
naive_rmse = np.sqrt(mean_squared_error(y_test_time, naive_predictions))
naive_r2 = r2_score(y_test_time, naive_predictions)


# ============================================================
# 13. MODEL COMPARISON
# ============================================================

model_comparison = pd.DataFrame({
    "Model": [
        "Random Forest (Random Split)",
        "Naive Previous-Value Baseline",
        "Random Forest (Time-Series Split)",
        "Gradient Boosting (Time-Series Split)"
    ],
    "MAE (Wh)": [
        mae,
        naive_mae,
        mae_time,
        mae_gb
    ],
    "RMSE (Wh)": [
        rmse,
        naive_rmse,
        rmse_time,
        rmse_gb
    ],
    "R2 Score": [
        r2,
        naive_r2,
        r2_time,
        r2_gb
    ]
})

print("\nMODEL COMPARISON")
print("================")
print(model_comparison.round(4).to_string(index=False))

model_comparison.to_csv(
    "model_comparison.csv",
    index=False
)


# ============================================================
# 14. ACTUAL VS PREDICTED VISUALIZATION
# ============================================================

plt.figure(figsize=(14, 5))

plt.plot(
    y_test_time.values[:300],
    label="Actual Energy Consumption"
)

plt.plot(
    y_pred_time[:300],
    label="Predicted Energy Consumption"
)

plt.xlabel("Test Samples")
plt.ylabel("Energy Consumption (Wh)")
plt.title("Actual vs Predicted Energy Consumption")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 15. ENERGY-SAVING RECOMMENDATION SYSTEM
# ============================================================

def energy_recommendation(energy, hour):
    recommendations = []

    if energy > upper_limit:
        recommendations.append(
            "High energy consumption detected. "
            "Check for unnecessary appliance usage or equipment running."
        )

    if 17 <= hour <= 20:
        recommendations.append(
            "Peak consumption period. Consider reducing non-essential "
            "appliance usage during this time."
        )

    if 0 <= hour <= 5:
        recommendations.append(
            "Low-demand period. Schedule flexible energy-intensive "
            "operations during this period when possible."
        )

    if not recommendations:
        recommendations.append(
            "Energy consumption is within the normal range. "
            "Continue monitoring usage."
        )

    return recommendations


sample_data = df[["date", "hour", "Appliances"]].tail(10).copy()

for _, row in sample_data.iterrows():
    recommendations = energy_recommendation(
        row["Appliances"],
        row["hour"]
    )

    print(
        f"\nTime: {row['date']} | "
        f"Energy: {row['Appliances']} Wh"
    )

    for recommendation in recommendations:
        print("Recommendation:", recommendation)


# ============================================================
# 16. SAVE FINAL PROJECT RESULTS
# ============================================================

hourly_results = (
    df.groupby("hour")["Appliances"]
    .mean()
    .reset_index()
)

hourly_results.columns = [
    "Hour",
    "Average_Energy_Wh"
]

anomaly_results = df[
    ["date", "Appliances", "hour", "is_anomaly"]
].copy()

anomaly_results.columns = [
    "Date",
    "Energy_Consumption_Wh",
    "Hour",
    "Is_Anomaly"
]

prediction_results = pd.DataFrame({
    "Actual_Energy_Wh": y_test_time.values,
    "Predicted_Energy_Wh": y_pred_time
})

hourly_results.to_csv(
    "hourly_energy_analysis.csv",
    index=False
)

anomaly_results.to_csv(
    "anomaly_detection_results.csv",
    index=False
)

prediction_results.to_csv(
    "energy_predictions.csv",
    index=False
)


# ============================================================
# 17. PROJECT SUMMARY
# ============================================================

summary = {
    "Total_Records": len(df),
    "Anomalies_Detected": anomaly_count,
    "Anomaly_Percentage": round(
        anomaly_count / len(df) * 100, 2
    ),
    "Peak_Hour": int(hourly_energy.idxmax()),
    "Peak_Average_Energy_Wh": round(
        hourly_energy.max(), 2
    ),
    "Lowest_Hour": int(hourly_energy.idxmin()),
    "Lowest_Average_Energy_Wh": round(
        hourly_energy.min(), 2
    ),
    "Baseline_R2": round(r2, 4),
    "Improved_R2": round(r2_time, 4),
    "Improved_MAE_Wh": round(mae_time, 2),
    "Improved_RMSE_Wh": round(rmse_time, 2)
}

summary_df = pd.DataFrame(
    summary.items(),
    columns=["Metric", "Value"]
)

summary_df.to_csv(
    "project_summary.csv",
    index=False
)

print("\n==========================================")
print(" INTELLIGENT ENERGY ANALYTICS DASHBOARD")
print("==========================================")
print(f"\nTotal Records Analyzed      : {len(df):,}")
print(f"Anomalies Detected         : {anomaly_count:,}")
print(
    f"Anomaly Percentage         : "
    f"{anomaly_count / len(df) * 100:.2f}%"
)
print(
    f"Peak Consumption Hour      : "
    f"{int(hourly_energy.idxmax()):02d}:00"
)
print(
    f"Peak Average Consumption   : "
    f"{hourly_energy.max():.2f} Wh"
)
print(
    f"Lowest Consumption Hour    : "
    f"{int(hourly_energy.idxmin()):02d}:00"
)
print(
    f"Lowest Average Consumption : "
    f"{hourly_energy.min():.2f} Wh"
)
print(f"Best R² Score               : {r2:.4f}")
print("Best Model                  : Random Forest (Random Split)")

print("\nResult files created:")
print("- hourly_energy_analysis.csv")
print("- anomaly_detection_results.csv")
print("- energy_predictions.csv")
print("- project_summary.csv")
print("- model_comparison.csv")
