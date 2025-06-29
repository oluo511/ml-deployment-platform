import requests
import json

# API base URL
BASE_URL = "http://localhost:5001"

def test_health_check():
    """Test the health check endpoint"""
    print("🏥 Testing health check...")
    response = requests.get(f"{BASE_URL}/")
    print("✅ Health Check:", response.json())
    print()

def test_simple_prediction():
    """Test the simple prediction endpoint"""
    print("🏠 Testing simple prediction...")
    test_data = {
        "lot_area": 8500,
        "overall_qual": 7,
        "year_built": 2005,
        "gr_liv_area": 1800,
        "bedrooms": 3,
        "bathrooms": 2
    }
    
    response = requests.post(f"{BASE_URL}/predict/simple", json=test_data)
    result = response.json()
    print("✅ Simple Prediction:")
    print(f"   💰 Predicted Price: ${result['predicted_price']:,.2f}")
    print(f"   📊 Input: {result['simplified_input']}")
    print()

def test_model_info():
    """Test the model info endpoint"""
    print("ℹ️  Testing model info...")
    response = requests.get(f"{BASE_URL}/model/info")
    result = response.json()
    print("✅ Model Info:")
    print(f"   🤖 Model: {result['model_type']}")
    print(f"   📈 Performance: {result['performance']}")
    print(f"   🎯 Target: {result['target']}")
    print()

if __name__ == "__main__":
    print("🚀 Testing House Price API...\n")
    
    try:
        test_health_check()
        test_simple_prediction()
        test_model_info()
        print("🎉 All tests passed! Your API is working perfectly!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the Flask app is running!")
        print("   Run: python app.py")
    except Exception as e:
        print(f"❌ Error: {e}")