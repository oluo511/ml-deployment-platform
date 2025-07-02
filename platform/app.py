import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px

# Page config
st.set_page_config(
    page_title="ML Deployment Platform",
    page_icon="🚀",
    layout="wide"
)

# Main title
st.title("🚀 ML Deployment Platform")
st.markdown("*From Jupyter notebook to production API in minutes*")

# Direct to Model Playground (no sidebar navigation)
page = "🏠 Model Playground"

if page == "🏠 Model Playground":
    st.header("🏠 House Price Predictor")
    st.markdown("Test our deployed house price model with real inputs!")
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("House Features")
        lot_area = st.number_input("Lot Area (sq ft)", min_value=1000, max_value=50000, value=8500, step=500)
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
        
    with col2:
        st.subheader("Property Details")
        year_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=2000, step=1)
        overall_qual = st.number_input("Overall Quality (1-10)", min_value=1, max_value=10, value=7, step=1)
        gr_liv_area = st.number_input("Living Area (sq ft)", min_value=500, max_value=5000, value=1800, step=100)
    
    # Predict button
    if st.button("🔮 Predict House Price", type="primary"):
        # Prepare data for API
        prediction_data = {
            "lot_area": int(lot_area),
            "bedrooms": int(bedrooms),
            "bathrooms": int(bathrooms),
            "year_built": int(year_built),
            "overall_qual": int(overall_qual),
            "gr_liv_area": int(gr_liv_area)
        }
        
        try:
            # Call your Flask API
            with st.spinner("🤖 Running ML model..."):
                response = requests.post(
                    "http://house-price-service:5001/predict/simple",
                    json=prediction_data
                )
            
            if response.status_code == 200:
                result = response.json()
                predicted_price = result['predicted_price']
                
                # Display results
                st.success("✅ Prediction Complete!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Price", f"${predicted_price:,.0f}")
                with col2:
                    price_per_sqft = predicted_price / gr_liv_area
                    st.metric("Price per Sq Ft", f"${price_per_sqft:.0f}")
                with col3:
                    st.metric("Model Used", "ElasticNet")
                
                # Show input summary
                st.subheader("Input Summary")
                input_df = pd.DataFrame([prediction_data])
                st.dataframe(input_df, use_container_width=True)
                
            else:
                st.error("❌ API Error: Could not get prediction")
                st.error(f"Status Code: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to model API. Make sure your Kubernetes deployment is running!")
            st.info("Check: kubectl get pods")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Footer
st.markdown("---")
st.markdown("*Kaggle Competition: GSB-544 House Prices | ElasticNet Regression | RMSE Score: 0.14993*")
st.markdown("*Built by Oscar Luo - ML Deployment Platform*")