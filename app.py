import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# List of features to ensure correct order when creating DataFrame
FEATURE_ORDER = ['Company_Name', 'Job_Title', 'Location', 'Job_Roles', 'Employment_Status', 'Salaries_Reported', 'Rating']

# --- 1. Load the Trained Model ---
try:
    MODEL_FILENAME = 'best_salary_model.pkl'
    model = joblib.load(MODEL_FILENAME)
    print(f"Model loaded successfully from {MODEL_FILENAME}.")
except FileNotFoundError:
    print(f"Error: {MODEL_FILENAME} not found. Did you run model_trainer.py?")
    model = None


# --- 2. Define Routes ---

@app.route('/')
def home():
    """Renders the main Home page."""
    return render_template('index.html')


@app.route('/predict_form')
def predict_form():
    """Renders the 'FILL THE DETAILS!' form page."""
    return render_template('Predict.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the HTML form."""
    if model is None:
        return render_template('Salresult.html', predicted_salary="Model Error: Model not available.")

    try:
        # to get form values
        company_name = request.form['company_name']
        job_title = request.form['job_title']
        location = request.form['location']
        job_roles = request.form['job_roles']
        employment_status = request.form['employment_status']
        
        # convert ONLY numeric fields
        salaries_reported = float(request.form['salaries_reported'])
        rating = float(request.form['rating'])

        # create DataFrame with exact column names (with spaces) in training order
        input_df = pd.DataFrame([[
            company_name,
            job_title,
            location,
            job_roles,
            employment_status,
            salaries_reported,
            rating
        ]], columns=[
            'Company Name',
            'Job Title',
            'Location',
            'Job Roles',
            'Employment Status',
            'Salaries Reported',
            'Rating'
        ])
        # Make prediction
        prediction_usd = model.predict(input_df)[0]
        
        #convert USD to INR (1 USD ≈ 83 INR )
        USD_TO_INR = 83.0
        prediction_inr = prediction_usd * USD_TO_INR
        
        # Format the salary nicely for display
        predicted_salary_str = f"₹{prediction_inr:,.0f}"

        # Render the result template
        return render_template('Salresult.html', predicted_salary=predicted_salary_str)

    except ValueError:
        return render_template('Salresult.html', predicted_salary="Input Error: Please ensure all fields are valid numbers.")
    except Exception as e:
        print(f"Error: {e}")
        return render_template('Salresult.html', predicted_salary=f"An unknown error occurred.")


# --- 3. Run the App ---
if __name__ == "__main__":
    # Use host='0.0.0.0' to allow access from other devices on the local network
    app.run(debug=True)
