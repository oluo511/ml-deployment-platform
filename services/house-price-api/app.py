from flask import Flask, request, jsonify  # Flask for web API, request for getting data, jsonify for JSON responses
import joblib  
import numpy as np  
import pandas as pd  
from datetime import datetime 
import os

# Create a Flask web application
# Think of this like creating a website that can receive requests and send responses
app = Flask(__name__)

# Load trained ML model 
model_path = 'models/house_price/saved_model/elastic_net_regression.pkl'
model = joblib.load(model_path)

@app.route('/', methods=['GET'])
def health_check():
    """
    Health check endpoint like a "are you alive?" ping
    When someone visits http://localhost:5001/ they get this response
    GET means this is just for viewing, not sending data
    """
    return jsonify({
        'status': 'healthy',
        'service': 'house-price-api',
        'timestamp': datetime.now().isoformat(),
        'model': 'elastic_net_regression'
    })

@app.route('/predict', methods=['POST'])
def predict_house_price():
    """
    Main prediction endpoint - takes house features as JSON, returns predicted price
    
    POST = sending data to us (house features) → we send back price prediction
    Uses JSON because it's the web standard and works with Streamlit
    """
    try:
        # Get the JSON data that someone sent to our API
        # This is like opening an envelope and reading the letter inside
        data = request.get_json()
        
        # Check if they actually sent us data
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Convert the JSON into a pandas DataFrame 
        # Why? Because that's what our model expects (same format as training)
        # The [data] creates a list with one dictionary, which becomes one row
        input_df = pd.DataFrame([data])
        
        # Make the prediction using our trained model
        # Remember: your model was trained on log-transformed prices (you used np.log1p)
        log_prediction = model.predict(input_df)
        
        # Convert back from log scale to actual dollars
        # np.expm1 is the inverse of np.log1p that you used in training
        predicted_price = np.expm1(log_prediction[0])
        
        # Send back the prediction as JSON
        return jsonify({
            'predicted_price': round(predicted_price, 2),  # Round to 2 decimal places
            'model': 'elastic_net_regression',
            'timestamp': datetime.now().isoformat(),
            'input_features': data  # Echo back what they sent us
        })
        
    except Exception as e:
        # If anything goes wrong, send back an error message
        return jsonify({
            'error': str(e),
            'message': 'Error making prediction'
        }), 500

@app.route('/predict/simple', methods=['POST'])
def predict_simple():
    """
    Simplified prediction with only 6 key features instead of all 23
    We auto-fill the rest with reasonable defaults. Great for demos!
    
    Input: {"lot_area": 9605, "overall_qual": 7, "year_built": 2000, 
            "gr_liv_area": 1218, "bedrooms": 3, "bathrooms": 2}
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Create a full feature set using the simple inputs + reasonable defaults
        # This is like filling out a form where most boxes have default values
        full_features = {
            "Lot Frontage": 70.0,  # Default median from your data
            "Lot Area": data.get("lot_area", 10000),  # Use their input or default
            "Street": "Pave",  # Most houses have paved streets
            "Neighborhood": "NAmes",  # Most common neighborhood in your data
            "Bldg Type": "1Fam",  # Single family home (most common)
            "House Style": "1Story",  # Most common style
            "Overall Qual": data.get("overall_qual", 5),  # Their input or average
            "Overall Cond": 5,  # Average condition
            "Year Built": data.get("year_built", 1980),  # Their input or default
            "Roof Style": "Gable",  # Most common roof type
            "Heating": "GasA",  # Gas heating (most common)
            "Central Air": "Y",  # Assume yes (most houses have it)
            "Electrical": "SBrkr",  # Standard electrical (most common)
            "Full Bath": data.get("bathrooms", 1),  # Their input
            "Half Bath": 0,  # Default to no half bath
            "Bedroom AbvGr": data.get("bedrooms", 3),  # Their input
            "TotRms AbvGrd": data.get("bedrooms", 3) + 3,  # Estimate: bedrooms + 3 other rooms
            "Gr Liv Area": data.get("gr_liv_area", 1500),  # Their input or default
            "Functional": "Typ",  # Typical functionality
            "Screen Porch": 0,  # Most houses don't have screen porches
            "Pool Area": 0,  # Most houses don't have pools
            "Yr Sold": 2023,  # Current year
            "Sale Type": "WD"  # Warranty deed (most common sale type)
        }
        
        # Convert to DataFrame and make prediction (same as above)
        input_df = pd.DataFrame([full_features])
        log_prediction = model.predict(input_df)
        predicted_price = np.expm1(log_prediction[0])
        
        return jsonify({
            'predicted_price': round(predicted_price, 2),
            'simplified_input': data,  # Show what they sent us
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Error making prediction'
        }), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    try:
        return jsonify({
            'model_type': 'ElasticNet Regression',
            'model_path': model_path,
            'features_required': [
                "Lot Frontage", "Lot Area", "Street", "Neighborhood", 
                "Bldg Type", "House Style", "Overall Qual", "Overall Cond",
                "Year Built", "Roof Style", "Heating", "Central Air", 
                "Electrical", "Full Bath", "Half Bath", "Bedroom AbvGr",
                "TotRms AbvGrd", "Gr Liv Area", "Functional", 
                "Screen Porch", "Pool Area", "Yr Sold", "Sale Type"
            ],
            'target': 'SalePrice (USD)',
            'performance': 'RMSE: 0.1506 (cross-validated)',
            'last_updated': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # This runs the Flask web server
    # debug=True means it will restart automatically when you change the code
    # host='0.0.0.0' means it accepts connections from anywhere (not just localhost)
    # port=5001 means it runs on http://localhost:5001
    app.run(debug=True, host='0.0.0.0', port=5001)