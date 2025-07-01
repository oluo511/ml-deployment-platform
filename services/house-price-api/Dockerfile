# Give me a virtual computer with Python 3.11 already installed
# slim meaning lightweight version
FROM python:3.11-slim 

# Create and set app folder as the root directory for project
WORKDIR /app

# Copying requirements.txt into app folder
COPY services/house-price-api/requirements.txt .

# Install dependencies and delete cache because the temporary files are only needed
# for installation and once libraries are installed, they are part of the image forever
# Images = everthing required for app to run like Python, dependencies, code
RUN pip install --no-cache-dir -r requirements.txt

# Copy the ML model files
# Goes up two levels, house-price-api -> services -> ml-deployment-platform
COPY models/house_price/saved_model/elastic_net_regression.pkl ./models/house_price/saved_model/
# Copy application code
COPY services/house-price-api/app.py .

# Tell Docker that this container will use port 5001
# 5000 is Flask default port
EXPOSE 5001

# Set environment variables (probably don't need this good practice)
# Creates two variables in container environment

# Allows Python to see and import from /app folder 
ENV PYTHONPATH=/app

# Tells Flask which file to run
ENV FLASK_APP=app.py

# Command to run when container starts
CMD ["python", "app.py"]