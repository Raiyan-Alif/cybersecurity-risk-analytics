# Cybersecurity Risk Analytics

This project analyses global cybersecurity incidents from 2015 to 2024 using Apache Spark and Spark MLlib.

The system applies K-Means clustering to create cybersecurity risk profiles and uses Linear Regression to predict incident resolution time. The Spark-trained Linear Regression model was exported into a portable JSON format and is served through FastAPI on Render. Predictions are accessed through an interactive Streamlit dashboard.

## Live Application Links

### FastAPI Backend

https://raiyan-alif-cybersecurity-risk-analytics.onrender.com

### FastAPI Health Check

https://raiyan-alif-cybersecurity-risk-analytics.onrender.com/health

### FastAPI Swagger Documentation

https://raiyan-alif-cybersecurity-risk-analytics.onrender.com/docs

### Streamlit Dashboard

https://cybersecurity-risk-analytics-wmd97djap9nztejrgkjy2m.streamlit.app/

## Main Project Files

- `main.py` — FastAPI backend and prediction endpoint
- `app.py` — Streamlit dashboard
- `portable_model.json` — exported model parameters
- `input_options.json` — valid categorical input options
- `requirements.txt` — required Python libraries
- `exports/` — dashboard and model-analysis CSV files

## Model Information

The final supervised model is Apache Spark MLlib Linear Regression.

The complete preprocessing workflow includes:

- StringIndexer
- OneHotEncoder
- VectorAssembler
- StandardScaler
- Linear Regression

The original Spark PipelineModel is preserved separately in the submitted model package.

## Running the Application Locally

### 1. Install the dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit dashboard

```bash
streamlit run app.py
```

The Streamlit application will normally open at:

```text
http://localhost:8501
```

By default, the application connects to the deployed FastAPI backend on Render.

### 3. Optional: Run FastAPI locally

To run the FastAPI backend locally:

```bash
python -m uvicorn main:app --reload
```

The local FastAPI service will be available at:

```text
http://127.0.0.1:8000
```

Swagger documentation will be available at:

```text
http://127.0.0.1:8000/docs
```

## Example Prediction

Example input:

```json
{
  "country": "China",
  "year": 2019,
  "attack_type": "Phishing",
  "target_industry": "Education",
  "financial_loss_million": 80.53,
  "number_of_affected_users": 773169,
  "attack_source": "Hacker Group",
  "security_vulnerability_type": "Unpatched Software",
  "defense_mechanism_used": "VPN"
}
```

Example output:

```json
{
  "predicted_resolution_time_hours": 38.47,
  "model_name": "Spark Linear Regression"
}
```

## Author

Md. Raiyan Alam Alif  
Student ID: CYS2302195
