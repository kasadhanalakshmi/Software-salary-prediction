import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# --- 1. Load Data ---
DATA_FILE = 'Salary_Dataset_with_Extra_Features.csv'

try:
    df = pd.read_csv(DATA_FILE)
    print(f" Data loaded successfully from {DATA_FILE}")
except FileNotFoundError:
    print(f" Error: {DATA_FILE} not found. Please place your CSV file in the project root.")
    exit()

# --- 2. Define Features (X) and Target (y) ---
# NOTE: Update these lists based on which features are truly numbers and which are text in your CSV.
# We are assuming Company_Name and Job_Title are the categorical (text) features causing the error.
CATEGORICAL_FEATURES = ['Company Name', 'Job Title','Location','Job Roles','Employment Status']
NUMERICAL_FEATURES = [  'Salaries Reported', 'Rating']
TARGET = 'Salary'

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

try:
    X = df[ALL_FEATURES]
    y = df[TARGET]
except KeyError as e:
    print(f" Error: Column {e} not found in the dataset. Check your spelling/capitalization.")
    exit()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Create Preprocessing Pipeline (The Fix) ---
preprocessor = ColumnTransformer(
    transformers=[
        # Apply OneHotEncoder to text features (like 'Xome')
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES),
        # Apply StandardScaler to numerical features
        ('num', StandardScaler(), NUMERICAL_FEATURES)
    ],
    remainder='passthrough' 
)


# --- 4. Define Models and Pipelines ---
models = {
    'Linear_Regression': Pipeline(steps=[('preprocessor', preprocessor),
                                        ('regressor', LinearRegression())]),

    'Random_Forest': Pipeline(steps=[('preprocessor', preprocessor),
                                    ('regressor', RandomForestRegressor(n_estimators=10, random_state=42))]
    )
}

# --- 5. Train, Evaluate, and Compare Models (The rest of the code is unchanged) ---

best_model = None
best_mae = float('inf')
best_model_name = ""
results = {}

print("\n--- Starting Model Training and Evaluation ---")

for name, model in models.items():
    # Train (This step now handles encoding the text 'Xome' and scaling the numbers)
    model.fit(X_train, y_train) 

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MAE': mae, 'R2': r2}

    print(f"\nModel: {name}")
    print(f"  Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"  R-squared (R2): {r2:.4f}")

    # Check for best model
    if mae < best_mae:
        best_mae = mae
        best_model = model
        best_model_name = name


# --- 6. Save the Best Model ---
MODEL_FILENAME = 'best_salary_model.pkl'
if best_model:
    joblib.dump(best_model, MODEL_FILENAME)
    print(f"\nBest model is '{best_model_name}' (MAE: ${best_mae:,.2f}).")
    print(f"Model saved as '{MODEL_FILENAME}'.")
else:
    print("\n Training failed. No model saved.")
