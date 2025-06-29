import requests
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def test_both_endpoints():
    """Compare accuracy between simple and full endpoints using training data"""
    
    # Load your TRAINING data (which has actual prices)
    try:
        train_data = pd.read_csv(r'data\train_new.csv')
        # Use last 10 rows as test data (ones you haven't seen much)
        test_data = train_data.tail(10)
        print(f"Using last 10 rows from training data for testing")
    except:
        print("❌ Could not load train_new.csv")
        return
    
    # Test with the sample
    test_sample = test_data
    actual_prices = test_sample['SalePrice'].tolist()  # Get actual prices
    
    simple_predictions = []
    full_predictions = []
    
    for idx, row in test_sample.iterrows():
        print(f"Testing row {idx}...")
        
        # Test simple endpoint (6 features)
        simple_data = {
            "lot_area": row.get("Lot Area", 10000),
            "bedrooms": row.get("Bedroom AbvGr", 3),
            "bathrooms": row.get("Full Bath", 2),
            "year_built": row.get("Year Built", 2000),
            "overall_qual": row.get("Overall Qual", 5),
            "gr_liv_area": row.get("Gr Liv Area", 1500)
        }
        
        try:
            response = requests.post("http://localhost:5001/predict/simple", json=simple_data)
            if response.status_code == 200:
                simple_pred = response.json()['predicted_price']
                simple_predictions.append(simple_pred)
            else:
                print(f"Simple API error for row {idx}")
                simple_predictions.append(None)
        except Exception as e:
            print(f"Simple API failed for row {idx}: {e}")
            simple_predictions.append(None)
        
        # Test full endpoint (all 23 features)
        full_data = {}
        for col in test_data.columns:
            if col not in ['PID', 'SalePrice']:  # Exclude PID and SalePrice
                full_data[col] = row[col] if pd.notna(row[col]) else 0
        
        try:
            response = requests.post("http://localhost:5001/predict", json=full_data)
            if response.status_code == 200:
                full_pred = response.json()['predicted_price']
                full_predictions.append(full_pred)
            else:
                print(f"Full API error for row {idx}")
                full_predictions.append(None)
        except Exception as e:
            print(f"Full API failed for row {idx}: {e}")
            full_predictions.append(None)
    
    # Compare predictions vs actual prices
    print("\n" + "="*70)
    print("PREDICTION COMPARISON vs ACTUAL PRICES")
    print("="*70)
    
    for i, (simple, full, actual) in enumerate(zip(simple_predictions, full_predictions, actual_prices)):
        if simple and full:
            simple_error = abs(simple - actual)
            full_error = abs(full - actual)
            diff_simple_full = abs(simple - full)
            diff_pct = (diff_simple_full / full) * 100
            
            print(f"Row {i}:")
            print(f"  Actual:   ${actual:,.0f}")
            print(f"  Simple:   ${simple:,.0f} (error: ${simple_error:,.0f})")
            print(f"  Full:     ${full:,.0f} (error: ${full_error:,.0f})")
            print(f"  Diff:     ${diff_simple_full:,.0f} ({diff_pct:.1f}%)")
            print()
    
    # Calculate metrics
    valid_data = [(s, f, a) for s, f, a in zip(simple_predictions, full_predictions, actual_prices) if s and f]
    if valid_data:
        simple_vals = [item[0] for item in valid_data]
        full_vals = [item[1] for item in valid_data]
        actual_vals = [item[2] for item in valid_data]
        
        # Errors vs actual prices
        simple_mae = np.mean([abs(s - a) for s, a in zip(simple_vals, actual_vals)])
        full_mae = np.mean([abs(f - a) for f, a in zip(full_vals, actual_vals)])
        
        # Difference between simple and full
        avg_diff = np.mean([abs(s - f) for s, f in zip(simple_vals, full_vals)])
        avg_diff_pct = np.mean([abs(s - f) / f * 100 for s, f in zip(simple_vals, full_vals)])
        
        print(f"📊 ACCURACY SUMMARY:")
        print(f"Simple endpoint MAE vs actual: ${simple_mae:,.0f}")
        print(f"Full endpoint MAE vs actual:   ${full_mae:,.0f}")
        print(f"Difference between endpoints:  ${avg_diff:,.0f} ({avg_diff_pct:.1f}%)")
        
        # Which is more accurate?
        if simple_mae < full_mae:
            print("😱 Simple endpoint is MORE accurate than full! (Unexpected)")
        else:
            accuracy_loss = ((simple_mae - full_mae) / full_mae) * 100
            print(f"📉 Simple endpoint is {accuracy_loss:.1f}% less accurate than full")
        
        if avg_diff_pct > 20:
            print("⚠️  HIGH DIFFERENCE - Simple model significantly different from full")
        elif avg_diff_pct > 10:
            print("⚠️  MODERATE DIFFERENCE - Some difference")
        else:
            print("✅ LOW DIFFERENCE - Simple model close to full")
    
    return simple_predictions, full_predictions

def compare_with_manual_test():
    """Test with one manual example to see the difference clearly"""
    print("\n" + "="*50)
    print("MANUAL TEST COMPARISON")
    print("="*50)
    
    # Create a test house
    manual_house = {
        "Lot Frontage": 80.0,
        "Lot Area": 9605,
        "Street": "Pave",
        "Neighborhood": "SawyerW",
        "Bldg Type": "1Fam",
        "House Style": "1Story",
        "Overall Qual": 7,
        "Overall Cond": 6,
        "Year Built": 2000,
        "Roof Style": "Gable",
        "Heating": "GasA",
        "Central Air": "Y",
        "Electrical": "SBrkr",
        "Full Bath": 2,
        "Half Bath": 1,
        "Bedroom AbvGr": 3,
        "TotRms AbvGrd": 6,
        "Gr Liv Area": 1800,
        "Functional": "Typ",
        "Screen Porch": 0,
        "Pool Area": 0,
        "Yr Sold": 2009,
        "Sale Type": "WD"
    }
    
    # Test simple (extracted features)
    simple_data = {
        "lot_area": 9605,
        "bedrooms": 3,
        "bathrooms": 2,
        "year_built": 2000,
        "overall_qual": 7,
        "gr_liv_area": 1800
    }
    
    # Call both endpoints
    try:
        simple_response = requests.post("http://localhost:5001/predict/simple", json=simple_data)
        full_response = requests.post("http://localhost:5001/predict", json=manual_house)
        
        if simple_response.status_code == 200 and full_response.status_code == 200:
            simple_price = simple_response.json()['predicted_price']
            full_price = full_response.json()['predicted_price']
            
            diff = abs(simple_price - full_price)
            diff_pct = (diff / full_price) * 100
            
            print(f"🏠 Test House:")
            print(f"Simple endpoint: ${simple_price:,.0f}")
            print(f"Full endpoint:   ${full_price:,.0f}")
            print(f"Difference:      ${diff:,.0f} ({diff_pct:.1f}%)")
            
            if diff_pct > 15:
                print("⚠️  Significant accuracy loss with simple endpoint")
            else:
                print("✅ Acceptable accuracy with simple endpoint")
                
        else:
            print("❌ API calls failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🧪 Testing Model Accuracy: Simple vs Full Endpoints")
    print("Make sure your Flask API is running on localhost:5001\n")
    
    # Test 1: Real data comparison
    test_both_endpoints()
    
    # Test 2: Manual example
    compare_with_manual_test()
    
    print("\n💡 Recommendations:")
    print("- If difference > 20%: Add more fields to simple form")
    print("- If difference < 10%: Simple form is good enough")
    print("- Consider offering both simple and advanced options")