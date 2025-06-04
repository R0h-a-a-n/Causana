# Causana

Causana is a modular, end-to-end system designed to analyze multivariate time series data and uncover temporal causal relationships. It enables users to submit datasets, perform causal inference, and visualize directed acyclic graphs (DAGs) that represent causal links across variables and time lags.

## Project Structure

The current implementation consists of three main components, each running as an independent service:

### 1. Spring Boot API Gateway

Handles HTTP requests from the frontend or API clients. Receives uploaded files and job configuration, then forwards requests to the Go-based job runner.

* Port: 8080
* Endpoint: `POST /submit-job`
* Fields: `file`, `method`, `lags`, `window`

### 2. Go Job Runner

Receives requests from the Spring Boot gateway, forwards the files and parameters to the Python causal engine, and returns the result to the gateway.

* Port: 8081
* Endpoint: `POST /run-job`

### 3. Python Causal Inference Engine

Implements basic Granger causality to compute causal relationships. Accepts CSV data and job parameters, processes the dataset, and returns a list of nodes and directed edges with associated weights and lags.

* Port: 8000
* Endpoint: `POST /granger`

## Features

* File upload and job submission via HTTP (Postman or frontend)
* Asynchronous processing pipeline from Java to Go to Python
* Causal graph output as a list of nodes and weighted edges
* Basic Granger causality support
* Configurable lag and window parameters

## Requirements

* Java 17 or higher
* Go 1.20 or higher
* Python 3.9 or higher
* Required Python packages: `pandas`, `numpy`, `statsmodels`, `fastapi`, `uvicorn`

## Local Development Setup

Start each service from its respective root directory:

### 1. Python Causal Inference Engine

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

### 2. Go Job Runner

```bash
go run main.go
```

### 3. Spring Boot Gateway

```bash
./mvnw spring-boot:run
```

## Testing the Pipeline

Use Postman or a similar client to send a `multipart/form-data` request to:

```
POST http://localhost:8080/submit-job
```

Form fields:

* `file`: CSV file (currently max 1 MB)
* `method`: `granger`
* `lags`: integer (e.g., 2)
* `window`: integer (e.g., 50)

A successful response will return a JSON object:

```json
{
  "nodes": ["X1", "X2", "X3"],
  "edges": [
    {"source": "X1", "target": "X2", "weight": 0.84, "lag": 1}
  ]
}
```

## Notes

* The current MVP supports basic Granger causality; more advanced algorithms will be integrated in future updates.
* Larger file support requires adjustment of server upload size limits.
* The frontend is under development and will provide an interactive DAG viewer and time series explorer.
