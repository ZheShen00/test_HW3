# MLDS-423-HW3 Cloud Type Classifier Web App

This is a Streamlit web application that allows users to classify cloud types using machine learning models. The app supports model version selection, single prediction input, and batch prediction via CSV upload.

## Features

- **Model Selection**: Choose between multiple versions of trained models (e.g., LogisticRegression, RandomForestClassifier).
- **Single Prediction**: Input three features manually and get an instant prediction.
- **Batch Prediction**: Upload a CSV file and receive predictions for multiple samples.
- **Error Handling**: User-friendly messages are displayed for model loading or input issues.
- **Logging**: All runtime events and exceptions are logged to `app.log`.
- **Unit Testing**: Streamlit interface and prediction functionality are covered by test cases.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/NUMLDS/423-2024-hw3-vqy2214.git
cd 423-2024-hw3-vqy2214
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Locally

```bash
streamlit run app.py
```

---

## Docker Support

### Build the Docker Image

```bash
docker build -t 423-hw3-vqy2214-streamlit-app .
```

### Run the Docker Container

```bash
# Visit http://localhost:8501/ after running this
docker run -p 8501:8501 423-hw3-vqy2214-streamlit-app
```


---

## Project Structure

```
.
├── app.py                  # Streamlit application
├── models/                 # Contains trained model files (.pkl)
|    ├── LogisticRegression.pkl
|    └── RandomForestClassifier.pkl
├── tests/
|    ├── test.csv           # test data for unit tests
|    └── test_app.py        # Unit tests
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker container definition
├── .gitignore              # Git ignore rules
└── data_sample.csv         # sample of data for batch prediction on website 
```

---

## Running Unit Tests

```bash
pytest tests/
```

Tests cover:

* Application loading
* Single prediction input and output
* Batch prediction CSV processing

---

## Required CSV Format

For batch predictions, your CSV file must contain the following columns:

* `log_entropy`
* `IR_norm_range`
* `entropy_x_contrast`

Example:

```csv
log_entropy,IR_norm_range,entropy_x_contrast
0.12,0.34,0.56
0.23,0.45,0.67
```

---

## Deployment on AWS

This application has been successfully deployed on **AWS ECS Fargate** with an **Application Load Balancer (ALB)** in front to route traffic.

You can visit the deployed Web App here:
[http://streamlit8501-393275156.us-east-1.elb.amazonaws.com/](http://streamlit8501-393275156.us-east-1.elb.amazonaws.com/)

**Access Restrictions**:
This website is only available from **specific IPv4 address ranges**, including:

* **Northwestern University** network ranges
* **My home IP range**

To access the app:

* Use **Northwestern VPN** or
* Be connected from one of the pre-approved IP ranges

This is done to ensure **deployment security** and avoid unsafe exposure to the public.

---

## AWS Security Group Configuration

To satisfy **deployment security requirements** while ensuring **Application Load Balancer (ALB)** health checks and client access work as expected, the following **inbound rules** were configured for the ECS Task Security Group:

| Port | Protocol | Source                | Purpose                          |
| ---- | -------- | --------------------- | -------------------------------- |
| 80   | TCP      | Approved IPv4 ranges  | For HTTP ALB front-end access    |
| 8501 | TCP      | ALB Security Group ID | For ALB health checks & routing  |
| 8501 | TCP      | Approved IPv4 ranges  | For direct IP-based access tests |

This configuration ensures:

* Public requests route through ALB on port 80
* ALB can internally communicate with the container on port 8501
* The service remains **private but functional**

---

## Performance Optimizations and Caching

To improve runtime performance and meet the caching requirements in the assignment, the application uses **Streamlit's caching decorators**:

### Cached Model Loading

The following function is wrapped with `@st.cache_resource` to avoid redundant loading of large model files:

```python
@st.cache_resource
def load_model(model_name):
    ...
```

---

## Deployment Architecture Summary

* **Dockerized app** pushed to **Amazon ECR**
* **ECS Fargate** service runs the container
* **Application Load Balancer** routes incoming traffic
* **Target Group** performs health checks on port 8501
* **Security Group** allows fine-grained access control
* Deployment verified to be secure, functional, and performant

---

## Author

Zhe Shen
MLDS 423: Cloud Engineering for Data Science
Spring 2025
