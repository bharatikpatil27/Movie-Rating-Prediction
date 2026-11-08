import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

print("MOVIE RATING PREDICTION PROJECT")
print("--------------------------------")

# Locate the dataset
csv_files = glob.glob("*.csv") + glob.glob("*/*.csv")

if len(csv_files) == 0:
    print("ERROR: CSV dataset not found!")
    print("Please place your CSV file inside the SystemTron folder.")
    raise SystemExit

file_path = csv_files[0]

print("Dataset found:", file_path)

# Load dataset
try:
    df = pd.read_csv(file_path, encoding="latin1")
except Exception as e:
    print("Dataset loading error:", e)
    raise SystemExit

print("\nDataset loaded successfully!")

# Display first five records
print("\nFIRST FIVE MOVIES:")
print(df.head())

# Dataset dimensions
print("\nDATASET SHAPE:")
print(df.shape)

# Column names
print("\nCOLUMN NAMES:")
print(df.columns.tolist())

# Dataset information
print("\nDATASET INFORMATION:")
df.info()

# Missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Statistical summary
print("\nSTATISTICAL SUMMARY:")
print(df.describe(include="all"))

print("\nDATASET ANALYSIS COMPLETED SUCCESSFULLY!")

# STEP 5: DATA CLEANING AND PREPROCESSING

print("\n--- DATA CLEANING STARTED ---")

# Remove duplicate records
df.drop_duplicates(inplace=True)

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Convert rating column to numeric
if "Rating" in df.columns:
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

    # Remove rows where rating is missing
    df.dropna(subset=["Rating"], inplace=True)

# Fill missing text values
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].fillna("Unknown")

# Fill missing numeric values except Rating
for col in df.select_dtypes(include=["number"]).columns:
    if col != "Rating":
        df[col] = df[col].fillna(df[col].median())

print("Data cleaning completed successfully!")

print("\nRemaining missing values:")
print(df.isnull().sum())

print("\nCleaned dataset shape:")
print(df.shape)

# Save cleaned dataset
df.to_csv("cleaned_movies.csv", index=False)

print("\nCleaned dataset saved successfully!")

# STEP 6: DATA VISUALIZATION

import os

print("\n--- DATA VISUALIZATION STARTED ---")

os.makedirs("charts", exist_ok=True)

# 1. Movie Rating Distribution
if "Rating" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Rating"].dropna(), bins=20, kde=True)
    plt.title("Movie Rating Distribution")
    plt.xlabel("Movie Rating")
    plt.ylabel("Number of Movies")
    plt.savefig("charts/rating_distribution.png")
    plt.show()

# 2. Top 10 Movie Genres
if "Genre" in df.columns:
    genre_counts = (
        df["Genre"].dropna()
        .str.split(",")
        .explode()
        .str.strip()
        .value_counts()
        .head(10)
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(x=genre_counts.index, y=genre_counts.values)
    plt.title("Top 10 Movie Genres")
    plt.xlabel("Genre")
    plt.ylabel("Number of Movies")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("charts/top_genres.png")
    plt.show()

# 3. Movie Ratings vs Votes
if "Rating" in df.columns and "Votes" in df.columns:
    plot_df = df[["Rating", "Votes"]].copy()
    plot_df["Votes"] = pd.to_numeric(
        plot_df["Votes"].astype(str).str.replace(",", ""),
        errors="coerce"
    )
    plot_df = plot_df.dropna()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=plot_df, x="Votes", y="Rating")
    plt.title("Movie Ratings vs Votes")
    plt.xlabel("Number of Votes")
    plt.ylabel("Movie Rating")
    plt.tight_layout()
    plt.savefig("charts/ratings_vs_votes.png")
    plt.show()

# 4. Average Rating by Genre
if "Rating" in df.columns and "Genre" in df.columns:
    genre_ratings = df[["Genre", "Rating"]].dropna().copy()
    genre_ratings["Genre"] = genre_ratings["Genre"].str.split(",")
    genre_ratings = genre_ratings.explode("Genre")
    genre_ratings["Genre"] = genre_ratings["Genre"].str.strip()

    avg_ratings = (
        genre_ratings.groupby("Genre")["Rating"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(x=avg_ratings.index, y=avg_ratings.values)
    plt.title("Average Rating by Genre")
    plt.xlabel("Genre")
    plt.ylabel("Average Rating")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("charts/average_rating_by_genre.png")
    plt.show()

print("\nData visualization completed successfully!")
print("All available graphs saved in the charts folder.")

# STEP 7: MOVIE RATING PREDICTION MODEL

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

print("\n--- MACHINE LEARNING MODEL STARTED ---")

# Select available features
possible_features = [
    "Genre", "Director", "Actor 1", "Actor 2", "Actor 3",
    "Year", "Duration", "Votes"
]

features = [col for col in possible_features if col in df.columns]

if "Rating" not in df.columns or len(features) == 0:
    raise ValueError("Required Rating or feature columns are missing.")

# Prepare data
model_df = df[features + ["Rating"]].copy()
model_df["Rating"] = pd.to_numeric(model_df["Rating"], errors="coerce")
model_df = model_df.dropna(subset=["Rating"])

# Convert numerical columns
numeric_cols = ["Year", "Duration", "Votes"]
numeric_features = [col for col in numeric_cols if col in features]
categorical_features = [col for col in features if col not in numeric_features]

for col in numeric_features:
    model_df[col] = pd.to_numeric(
        model_df[col].astype(str).str.replace(",", "", regex=False)
        .str.extract(r"(\d+\.?\d*)")[0],
        errors="coerce"
    )

X = model_df[features]
y = model_df["Rating"]

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=5,
        max_categories=50
    ))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Create regression model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict ratings
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nMODEL TRAINED SUCCESSFULLY!")

print("\nMODEL EVALUATION:")
print("Mean Absolute Error:", round(mae, 3))
print("Mean Squared Error:", round(mse, 3))
print("Root Mean Squared Error:", round(rmse, 3))
print("R2 Score:", round(r2, 3))

# Save trained model
joblib.dump(model, "movie_rating_model.pkl")

print("\nTrained model saved successfully!")

# Display sample predictions
results = pd.DataFrame({
    "Actual Rating": y_test.values[:10],
    "Predicted Rating": y_pred[:10]
})

print("\nSAMPLE MOVIE RATING PREDICTIONS:")
print(results)

results.to_csv("prediction_results.csv", index=False)

print("\nMOVIE RATING PREDICTION COMPLETED SUCCESSFULLY!")

# STEP 8: MODEL PERFORMANCE VISUALIZATION

print("\n--- MODEL PERFORMANCE VISUALIZATION ---")

# Actual vs Predicted Ratings
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Movie Ratings")
plt.ylabel("Predicted Movie Ratings")
plt.title("Actual vs Predicted Movie Ratings")
plt.tight_layout()
plt.savefig("charts/actual_vs_predicted.png")
plt.show()

# Prediction Error Distribution
errors = y_test.values - y_pred

plt.figure(figsize=(8, 5))
sns.histplot(errors, bins=20, kde=True)
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Prediction Error Distribution")
plt.tight_layout()
plt.savefig("charts/prediction_errors.png")
plt.show()

print("\nPerformance graphs saved successfully!")
