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

# Sidebar for navigation
st.sidebar.title("Platform Menu")
page = st.sidebar.selectbox(
    "Choose a section:",
    ["🏠 Model Playground", "📊 Deployed Models", "⚙️ Deploy New Model"]
)

if page == "🏠 Model Playground":
    st.header("🏠 House Price Predictor")
    st.markdown("Test our deployed house price model with real inputs!")
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("House Features")
        lot_area = st.number_input("Lot Area (sq ft)", min_value=1000, max_value=50000, value=8500)
        bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6], index=2)
        bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4, 5], index=1)
        
    with col2:
        st.subheader("Property Details")
        year_built = st.slider("Year Built", 1900, 2025, 2000)
        overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 7)
        gr_liv_area = st.number_input("Living Area (sq ft)", min_value=500, max_value=5000, value=1800)
    
    # Predict button
    if st.button("🔮 Predict House Price", type="primary"):
        # Prepare data for API
        prediction_data = {
            "lot_area": lot_area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "year_built": year_built,
            "overall_qual": overall_qual,
            "gr_liv_area": gr_liv_area
        }
        
        try:
            # Call your Flask API
            with st.spinner("🤖 Running ML model..."):
                response = requests.post(
                    "http://localhost:5001/predict/simple",
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
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to model API. Make sure Flask app is running!")
        except Exception as e:
            st.error(f"❌ Error: {e}")

elif page == "📊 Deployed Models":
    st.header("📊 Deployed Models")
    st.markdown("Monitor your live ML models")
    
    # Mock model status (in reality, this would query your APIs)
    models_data = {
        "Model": ["House Price Predictor", "Political Classifier"],
        "Status": ["🟢 Healthy", "🟢 Healthy"],
        "Requests Today": [47, 23],
        "Avg Response Time": ["120ms", "95ms"],
        "Accuracy": ["RMSE: 0.15", "64.4%"],
        "Deployed": ["2 hours ago", "Not deployed"]
    }
    
    models_df = pd.DataFrame(models_data)
    st.dataframe(models_df, use_container_width=True)
    
    # Performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("API Requests Over Time")
        # Mock time series data
        time_data = pd.DataFrame({
            'Hour': range(24),
            'Requests': [5, 3, 1, 0, 2, 8, 15, 25, 30, 28, 32, 35, 40, 38, 42, 45, 48, 50, 45, 35, 25, 18, 12, 8]
        })
        fig = px.line(time_data, x='Hour', y='Requests', title="Requests per Hour")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Model Performance")
        perf_data = pd.DataFrame({
            'Model': ['House Price', 'Political'],
            'Accuracy': [85, 64]
        })
        fig = px.bar(perf_data, x='Model', y='Accuracy', title="Model Accuracy %")
        st.plotly_chart(fig, use_container_width=True)

elif page == "⚙️ Deploy New Model":
    st.header("⚙️ Deploy New Model")
    st.markdown("Upload your trained model and deploy it as an API!")
    
    # Model upload section
    st.subheader("1. Upload Model")
    uploaded_file = st.file_uploader(
        "Choose a .pkl file", 
        type=['pkl'],
        help="Upload your trained scikit-learn model"
    )
    
    if uploaded_file:
        st.success(f"✅ Model uploaded: {uploaded_file.name}")
    
    # Model configuration
    st.subheader("2. Configure Deployment")
    col1, col2 = st.columns(2)
    
    with col1:
        model_name = st.text_input("Model Name", "my-awesome-model")
        model_type = st.selectbox("Model Type", ["Regression", "Classification"])
        
    with col2:
        api_port = st.number_input("API Port", min_value=5000, max_value=9999, value=5002)
        replicas = st.slider("Number of Replicas", 1, 5, 1)
    
    # Feature configuration
    st.subheader("3. Define Input Features")
    st.markdown("*What features does your model expect?*")
    
    feature_names = st.text_area(
        "Feature Names (one per line)",
        "feature_1\nfeature_2\nfeature_3",
        height=100
    )
    
    # Deploy button
    if st.button("🚀 Deploy Model", type="primary"):
        if uploaded_file and model_name:
            with st.spinner("🔧 Creating container and deploying..."):
                # This would actually:
                # 1. Save the uploaded model
                # 2. Generate Flask API code
                # 3. Create Dockerfile
                # 4. Build Docker image
                # 5. Deploy to Kubernetes
                import time
                time.sleep(3)  # Simulate deployment time
                
            st.success("🎉 Model deployed successfully!")
            st.balloons()
            
            # Show deployment details
            st.subheader("Deployment Details")
            st.code(f"""
API Endpoint: http://localhost:{api_port}/predict
Model: {model_name}
Status: Running
Container: ml-platform/{model_name}:latest
            """)
            
            # Show sample API call
            st.subheader("Sample API Call")
            st.code(f"""
curl -X POST http://localhost:{api_port}/predict \\
  -H "Content-Type: application/json" \\
  -d '{{"feature_1": 123, "feature_2": 456}}'
            """)
        else:
            st.error("Please upload a model file and provide a name")

# Footer
st.markdown("---")
st.markdown("*Built with ❤️ by Oscar Luo - ML Deployment Platform*")