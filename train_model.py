import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import joblib
import pickle
import os

# Dataset path
csv_path = "student dropout (1).csv"

# Load dataset
print("Loading dataset...")
df = pd.read_csv(csv_path)

# Display dataset info
print(f"Original dataset shape: {df.shape}")
print(f"\nMissing values before preprocessing:\n{df.isnull().sum()}")

# ===== DATA PREPROCESSING =====
print("\n" + "=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

# 1. Remove duplicate rows
print("\n1. Removing duplicate rows...")
initial_rows = len(df)
df = df.drop_duplicates()
print(f"   Removed {initial_rows - len(df)} duplicate rows")

# 2. Identify categorical and numerical columns (before preprocessing)
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Remove target column from preprocessing
if "Dropped_Out" in categorical_cols:
    categorical_cols.remove("Dropped_Out")

print(f"\n2. Initial columns:")
print(f"   Categorical: {categorical_cols}")
print(f"   Numerical: {numerical_cols}")

# 3. Remove columns with too many missing values
print("\n3. Removing columns with excessive missing values (>50%)...")
missing_threshold_cols = 0.5
initial_cols = len(df.columns)
missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
print(f"   Top missing % columns:\n{missing_pct[missing_pct > 30]}")
cols_with_high_missing = df.isnull().sum() / len(df) > missing_threshold_cols
print(f"   Columns with >50% missing: {df.columns[cols_with_high_missing].tolist()}")
df = df.loc[:, ~cols_with_high_missing]

# Update categorical and numerical columns based on remaining columns
categorical_cols = [col for col in categorical_cols if col in df.columns]
numerical_cols = [col for col in numerical_cols if col in df.columns]
print(
    f"   Removed {initial_cols - len(df.columns)} columns, Remaining: {len(df.columns)}"
)

# 4. Convert likely numeric columns to numeric type

# Try to convert columns that look like numbers to numeric types
likely_numeric = [
    "Age",
    "Mother_Education",
    "Father_Education",
    "Travel_Time",
    "Study_Time",
    "Number_of_Failures",
    "Family_Relationship",
    "Free_Time",
    "Going_Out",
    "Weekend_Alcohol_Consumption",
    "Weekday_Alcohol_Consumption",
    "Health_Status",
    "Number_of_Absences",
    "Grade_1",
    "Grade_2",
    "Final_Grade",
    "Family_Size",
]

for col in likely_numeric:
    if col in df.columns and col in categorical_cols:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            categorical_cols.remove(col)
            if col not in numerical_cols:
                numerical_cols.append(col)
        except:
            pass

print(f"\n4. Categorical columns: {categorical_cols}")
print(f"   Numerical columns: {numerical_cols}")

# 5. Handle invalid categorical values and convert boolean strings
print("\n5. Cleaning invalid categorical values...")
for col in categorical_cols:
    # First convert to string and strip whitespace
    df[col] = df[col].astype(str).str.strip()
    # Replace invalid values with NaN
    df[col] = df[col].replace(
        ["nan", "None", "INVALID", "invalid", "error", "unknown", "XYZ", "NaN"], np.nan
    )
    print(f"   {col}: Cleaned")

# 6. Remove rows with too many missing values (>40% missing)
print("\n6. Removing rows with excessive missing values (>40%)...")
missing_threshold = 0.4
initial_rows = len(df)
df = df.loc[df.isnull().sum(axis=1) / len(df.columns) < missing_threshold]
print(f"   Removed {initial_rows - len(df)} rows with >40% missing values")
print(f"   Dataset shape after removal: {df.shape}")

# 7. Handle missing values more aggressively
print("\n7. Filling all remaining missing values...")
# For numerical columns, fill with median
for col in numerical_cols:
    if col in df.columns:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            if pd.notna(median_val):
                df[col].fillna(median_val, inplace=True)
            else:
                df[col].fillna(
                    df[col].mode()[0] if len(df[col].mode()) > 0 else 0, inplace=True
                )

# For categorical columns, fill with mode
for col in categorical_cols:
    if col in df.columns:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown"
            df[col].fillna(mode_val, inplace=True)

# Fill any remaining NaN with 0 for numerical and "Unknown" for categorical
for col in df.columns:
    if df[col].dtype in ["float64", "int64"]:
        df[col].fillna(0, inplace=True)
    else:
        df[col].fillna("Unknown", inplace=True)

print(f"   Final NaN count: {df.isnull().sum().sum()}")

# 9. Handle target variable
print("\n9. Handling target variable...")
if "Dropped_Out" in df.columns:
    # Handle NaN in target
    missing_count = df["Dropped_Out"].isnull().sum()
    if missing_count > 0:
        mode_val = (
            df["Dropped_Out"].mode()[0] if len(df["Dropped_Out"].mode()) > 0 else 0
        )
        df["Dropped_Out"].fillna(mode_val, inplace=True)
        print(f"   Filled {missing_count} missing values in target with mode")
    print(f"   Final target value counts:\n{df['Dropped_Out'].value_counts()}")

# 10. Remove outliers in numerical columns (using IQR method)
print("\n10. Handling outliers in numerical columns...")
for col in numerical_cols:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Cap outliers instead of removing
        df[col] = df[col].clip(lower_bound, upper_bound)
        print(
            f"   {col}: Capped outliers (bounds: [{lower_bound:.2f}, {upper_bound:.2f}])"
        )

# 11. Validate data consistency
print("\n11. Final data validation...")
print(f"   Final dataset shape: {df.shape}")
print(f"   Missing values after preprocessing:\n{df.isnull().sum()}")

print("\nDataset after preprocessing:")
print(f"Dropout distribution:\n{df['Dropped_Out'].value_counts()}")

# Separate features and target
X = df.drop("Dropped_Out", axis=1)

# Convert target variable to binary (0 = didn't drop out, 1 = dropped out)
# Handle multiple possible values
target_mapping = {
    "True": 1,
    "true": 1,
    "1": 1,
    1: 1,
    "False": 0,
    "false": 0,
    "0": 0,
    0: 0,
    "Yes": 1,
    "yes": 1,
    "YES": 1,
    "No": 0,
    "no": 0,
    "NO": 0,
}

y = df["Dropped_Out"].copy()

# Remove rows with invalid target values (maybe, unknown, nan)
invalid_targets = ["maybe", "unknown", "nan", "None"]
mask = ~y.astype(str).str.lower().isin(invalid_targets) & y.notna()
y_filtered = y[mask]
X_filtered = X[mask]

# Replace NaN with 0 (assume didn't drop out if value is unknown)
y_filtered = y_filtered.fillna(0)

# Map values to binary
y_binary = y_filtered.astype(str).map(target_mapping)

# Handle any unmapped values
y = y_binary.fillna(0).astype(int)
X = X_filtered

# Encode categorical variables
print("\nEncoding categorical variables...")
label_encoders = {}
X_encoded = X.copy()

for col in categorical_cols:
    # Convert to string to ensure consistency before encoding
    X_encoded[col] = X[col].astype(str)
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col])
    label_encoders[col] = le
    print(f"  {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Standardize numerical features
print("\nStandardizing numerical features...")
scaler = StandardScaler()
if len(numerical_cols) > 0:
    print(f"   Scaling {len(numerical_cols)} numerical columns: {numerical_cols}")
    # Extract numerical data
    X_numerical = X_encoded[numerical_cols].values
    print(f"   X_numerical shape: {X_numerical.shape}, dtype: {X_numerical.dtype}")
    # Fit and transform
    X_scaled = scaler.fit_transform(X_numerical)
    print(f"   After fit_transform - Scaler has mean_: {hasattr(scaler, 'mean_')}")
    if hasattr(scaler, "mean_"):
        print(
            f"   Scaler mean_ shape: {scaler.mean_.shape}, values: {scaler.mean_[:3]}"
        )
    # Assign back
    X_encoded[numerical_cols] = X_scaled
    print(f"   Scaling complete")
else:
    print("   No numerical columns to scale")

# Split data into train and test sets
# Split data into train and test sets
print("\nPreparing data for training...")
print(f"Total rows before train/test split: {len(df)}")

# Final check for any remaining NaN values
if df.isnull().sum().sum() > 0:
    print(
        f"WARNING: Still have {df.isnull().sum().sum()} NaN values, dropping rows with NaN..."
    )
    df = df.dropna()
    print(f"Rows after NaN removal: {len(df)}")

print(f"Final dataset shape: {df.shape}")

# Ensure X_encoded has no NaN values
print("\nFinal NaN check before training...")
print(f"X_encoded NaN count before filling: {X_encoded.isnull().sum().sum()}")
# Fill any remaining NaNs with column medians or 0
for col in X_encoded.columns:
    if X_encoded[col].isnull().any():
        if X_encoded[col].dtype in ["float64", "int64"]:
            X_encoded[col].fillna(X_encoded[col].median(), inplace=True)
        else:
            X_encoded[col].fillna("Unknown", inplace=True)
# Final catch-all
X_encoded = X_encoded.fillna(0)
print(f"X_encoded NaN count after filling: {X_encoded.isnull().sum().sum()}")

print("\nSplitting data into train (80%) and test (20%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Train multiple models
print("\n" + "=" * 60)
print("TRAINING MODELS")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# Select Logistic Regression as the model
print("\n" + "=" * 60)
print("MODEL SELECTION")
print("=" * 60)

best_model_name = "Logistic Regression"
best_model = results[best_model_name]["model"]

print(f"\nSelected Model: {best_model_name}")
print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"Precision: {results[best_model_name]['precision']:.4f}")
print(f"Recall: {results[best_model_name]['recall']:.4f}")
print(f"F1-Score: {results[best_model_name]['f1']:.4f}")

# Create ml_models directory if it doesn't exist
ml_models_dir = "ml_models"
if not os.path.exists(ml_models_dir):
    os.makedirs(ml_models_dir)
    print(f"Created '{ml_models_dir}' directory")

# Save the best model
model_path = os.path.join(ml_models_dir, "dropout_model.joblib")
joblib.dump(best_model, model_path)
print(f"\nModel saved to: {model_path}")

# Save label encoders
encoders_path = os.path.join(ml_models_dir, "label_encoders.joblib")
joblib.dump(label_encoders, encoders_path)
print(f"Label encoders saved to: {encoders_path}")

# Save scaler using pickle for better serialization
print("\nSaving scaler using pickle...")
scaler_path = os.path.join(ml_models_dir, "scaler.pkl")
try:
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_path}")
except Exception as e:
    print(f"ERROR saving scaler: {e}")

# Save feature names
feature_names_path = os.path.join(ml_models_dir, "feature_names.joblib")
feature_names = X_encoded.columns.tolist()
joblib.dump(feature_names, feature_names_path)
print(f"Feature names saved to: {feature_names_path}")

# Save categorical and numerical column info with encoder metadata
info_path = os.path.join(ml_models_dir, "column_info.joblib")
encoder_metadata = {}
for col, le in label_encoders.items():
    encoder_metadata[col] = {
        "known_values": list(le.classes_),
        "n_classes": len(le.classes_),
        "default_encoding": 0,  # Use 0 for unknown values
    }

column_info = {
    "categorical": categorical_cols,
    "numerical": numerical_cols,
    "encoder_metadata": encoder_metadata,
}
joblib.dump(column_info, info_path)
print(f"Column info saved to: {info_path}")

print("\nEncoder metadata:")
for col, metadata in encoder_metadata.items():
    print(
        f"  {col}: {metadata['n_classes']} classes, known values: {metadata['known_values'][:5]}{'...' if metadata['n_classes'] > 5 else ''}"
    )

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)
