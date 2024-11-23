from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from scipy.optimize import linprog
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Configure upload folder
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load renewable energy data
def load_renewable_data():
    """Load historical renewable energy data, if available."""
    try:
        file_path = './uploads/renewable_data.csv'
        if not os.path.exists(file_path):
            return None  # Return None if no file is available

        renewable_data = pd.read_csv(file_path)
        if 'datetime' not in renewable_data.columns or 'renewable_energy' not in renewable_data.columns:
            raise ValueError("Renewable data must contain 'datetime' and 'renewable_energy' columns.")
        
        renewable_data['datetime'] = pd.to_datetime(renewable_data['datetime'], errors='coerce')
        if renewable_data['datetime'].isnull().any():
            raise ValueError("Invalid 'datetime' values in renewable data.")
        
        return renewable_data
    except Exception as e:
        print(f"Error loading renewable data: {e}")
        return None

# Preprocess uploaded data
def preprocess_data(file_path):
    """Preprocess the uploaded data for optimization."""
    try:
        data = pd.read_csv(file_path)
        if 'datetime' not in data.columns or 'electricity_consumption' not in data.columns:
            raise ValueError("File must contain 'datetime' and 'electricity_consumption' columns.")
        
        data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
        if data['datetime'].isnull().any():
            raise ValueError("Invalid 'datetime' values in file.")
        
        # Extract time-related features for further analysis
        data['hour'] = data['datetime'].dt.hour
        data['day'] = data['datetime'].dt.day
        data['month'] = data['datetime'].dt.month
        return data
    except Exception as e:
        raise ValueError(f"Data preprocessing failed: {e}")

# Predict future demand using Exponential Smoothing
def predict_demand(demand_data):
    """Predict future electricity demand using time series forecasting."""
    try:
        model = ExponentialSmoothing(
            demand_data, 
            trend="add", 
            seasonal="add", 
            seasonal_periods=24, 
            use_boxcox=False
        )
        model_fit = model.fit()
        return model_fit.forecast(steps=len(demand_data))
    except Exception as e:
        print(f"Prediction failed: {e}")
        return demand_data  # Fallback to current demand if prediction fails

# Optimize energy supply with linear programming
def optimize_energy(data, renewable_data=None):
    """Optimize energy supply using linear programming."""
    demand = data['electricity_consumption'].values
    
    # Use renewable data or simulate if unavailable
    renewable_energy = (
        np.interp(
            data['datetime'].view(int), 
            renewable_data['datetime'].view(int), 
            renewable_data['renewable_energy'].values
        ) if renewable_data is not None else np.random.uniform(100, 300, len(demand))
    )
    
    predicted_demand = predict_demand(demand)
    
    # Define cost and constraints
    energy_cost = 0.05
    carbon_footprint_rate = 0.2
    c = np.abs(demand - renewable_energy) + (energy_cost * predicted_demand) + (
        carbon_footprint_rate * np.maximum(predicted_demand - renewable_energy, 0)
    )
    
    bounds = [
        (max(renewable_energy[i] * 0.8, predicted_demand[i] * 0.9), predicted_demand[i] * 1.2)
        for i in range(len(predicted_demand))
    ]
    
    try:
        # Solve the optimization problem
        result = linprog(c, bounds=bounds, method='highs')
        if result.success:
            optimized_supply = result.x
            total_cost = np.sum(energy_cost * optimized_supply)
            total_carbon_footprint = np.sum(carbon_footprint_rate * np.maximum(optimized_supply - renewable_energy, 0))
            return optimized_supply, renewable_energy, total_cost, total_carbon_footprint
        
    except Exception as e:
        print(f"Optimization failed: {e}")
    
    # Fallback: return default strategy
    return demand * 1.1, renewable_energy, np.sum(energy_cost * demand * 1.1), np.sum(carbon_footprint_rate * (demand * 1.1 - renewable_energy))

# Routes
@app.route('/')
def welcome():
    """Welcome page."""
    return render_template('welcome.html')

@app.route('/index')
def index():
    """Main index page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """Handle file upload and return optimization results."""
    if 'file' not in request.files:
        return "No file part in the request", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if not allowed_file(file.filename):
        return "Invalid file type. Only CSV files are allowed.", 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    try:
        # Process uploaded data and optimize
        data = preprocess_data(file_path)
        renewable_data = load_renewable_data()
        optimized_supply, renewables, total_cost, total_carbon_footprint = optimize_energy(data, renewable_data)
        
        return jsonify({
            "optimized_supply": optimized_supply.tolist(),
            "renewables": renewables.tolist(),
            "demand": data['electricity_consumption'].tolist(),
            "total_cost": total_cost,
            "total_carbon_footprint": total_carbon_footprint
        })
    except Exception as e:
        print(f"Error processing file: {e}")
        return f"Error processing file: {str(e)}", 500

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
