# Causana: Temporal Causal Relationship Explorer

**Causana** is a modular, end-to-end system for analyzing **multivariate time series data** to uncover **temporal causal relationships**. Users can upload datasets, run causal inference algorithms, and visualize the results as **Directed Acyclic Graphs (DAGs)** showing causality across variables and lags.

---

## Project Structure

The **MVP** consists of three independently running services:

### 1. Spring Boot API Gateway

* **Role:** Receives file uploads and job configs, then forwards to the Go job runner.
* **Port:** `8080`
* **Endpoint:** `POST /submit-job`
* **Form Fields:** `file`, `method`, `lags`, `window`

### 2. Go Job Runner

* **Role:** Acts as a bridge between the gateway and the Python inference engine.
* **Port:** `8081`
* **Endpoint:** `POST /run-job`

### 3. Python Causal Inference Engine

* **Role:** Processes CSV data using **Granger causality**, returning nodes and directed edges with weights and lags.
* **Port:** `8000`
* **Endpoint:** `POST /granger`

---

## Features

* File upload and job submission via HTTP (Postman or UI)
* Modular architecture: Java → Go → Python
* Outputs causal DAG: list of nodes + weighted, lagged edges
* Basic Granger causality support (MVP)
* Configurable lags and window sizes

---

## Requirements

| Tool   | Version |
| ------ | ------- |
| Java   | 17+     |
| Go     | 1.20+   |
| Python | 3.9+    |

### Python Dependencies

```bash
pip install pandas numpy statsmodels fastapi uvicorn
```

---

## Local Development Setup

Start each service from its root folder:

### 1. Python Causal Engine

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

### 2. Go Job Runner

```bash
go run main.go
```

### 3. Spring Boot API Gateway

```bash
./mvnw spring-boot:run
```

---

## Testing the Pipeline (via Postman)

**Endpoint:**

```
POST http://localhost:8080/submit-job
```

**Form-Data Fields:**

* `file`: CSV file (max 1MB)
* `method`: `granger`
* `lags`: integer (e.g., `2`)
* `window`: integer (e.g., `50`)

**Sample JSON Response:**

```json
{
  "nodes": ["X1", "X2", "X3"],
  "edges": [
    { "source": "X1", "target": "X2", "weight": 0.84, "lag": 1 }
  ]
}
```

---

## Notes

* CSV size restrictions may apply (max 1MB in current setup).
* Some responses may be mock-based in MVP mode.
* Full statistical causality computation will be integrated in future versions.
* Frontend (React-based) is under active development and will support interactive DAG visualizations.
