# GovTech ML Risk Scoring Platform on GCP (Kubernetes-Based)

## Project Overview
This project focuses on building a production-grade, secure, and scalable Machine Learning (ML) platform on Google Cloud Platform (GCP) and Kubernetes. The primary objective is to simulate a risk-scoring system for public service claims.

The platform is designed with a focus on **Infrastructure as Code (IaC)**, **Security-first architecture**, and a **Fully reproducible ML lifecycle** to meet EU-standard engineering requirements.

## Problem Statement
Public service systems often face risks from anomalous or fraudulent claims that require rapid validation. This project addresses these challenges by providing an infrastructure capable of:
- Automated and distributed model training using Kubernetes Jobs.
- Scalable inference services using FastAPI hosted on GKE.
- Continuous monitoring for data drift to trigger automated retraining.

## System Architecture
The platform follows an integrated end-to-end lifecycle:
- **Data Layer:** Synthetic dataset generation and storage in Google Cloud Storage.
- **Orchestration Layer:** Execution of distributed training via Kubernetes Jobs on GKE.
- **Model Registry:** Versioning and artifact management using MLflow and Artifact Registry.
- **Serving Layer:** Model deployment as a FastAPI service with Horizontal Pod Autoscaling (HPA).
- **Monitoring Layer:** System and application observability via Prometheus and Grafana.

## Tech Stack
- **Cloud Provider:** Google Cloud Platform (GCP).
- **Infrastructure:** Kubernetes (GKE), Terraform (IaC), Docker.
- **ML Framework:** Python, PyTorch, MLflow.
- **CI/CD:** GitHub Actions.
- **Observability:** Prometheus, Grafana, Cloud Monitoring.

## Dataset Design
The synthetic dataset simulates public service claim risk profiles with the following specifications:
- **Target:** Binary classification (0 = low risk, 1 = high risk).
- **Primary Features:** `claim_amount`, `claim_frequency_30d`, `region_code`, `service_type`, `historical_flag_rate`, and `anomaly_score_rule_based`.
- **Characteristics:** 150,000 records with an injected fraud ratio of 3-7% based on specific business rules .

## ML Model Specification
The core model utilizes a Feedforward Neural Network architecture optimized for computational efficiency:
- **Architecture**: Input → Dense(64) → ReLU → Dense(32) → ReLU → Dense(1) → Sigmoid.
- **Loss Function**: Binary Cross Entropy.
- **Optimizer**: Adam.

## Infrastructure Design Principles
- **Least Privilege IAM:** Access separation using dedicated service accounts for training, inference, and CI/CD.
- **Environment Separation:** Workload isolation using Kubernetes Namespaces (`ml-dev` and `ml-prod`).
- **Reproducibility:** The entire infrastructure is defined through modular Terraform configurations.

## Getting Started

### Prerequisites
- Python 3.9+
- Docker
- Google Cloud SDK (gcloud)
- Terraform

### Local Development (Day 1)
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Generate dataset:
    ```
    python src/data/generator.py
    ```
3. Verify data integrity:
    ```
    python src/data/verify_data.py
    ```
4. Train baseline model:
    ```
    python src/models/train_baseline.py
    ```

## Roadmap

### Phase 1: Foundation (Completed)
- [x] Synthetic dataset generation with risk patterns.
- [x] Baseline model development using PyTorch.
- [x] Implementation of data verification and model smoke tests.

### Phase 2: Containerization & Cloud Infrastructure (Current)
- [ ] Dockerization of training and inference scripts.
- [ ] Provisioning of GKE cluster and VPC using Terraform.
- [ ] Setup of Artifact Registry and MLflow.

### Phase 3: Automation & Observability
- [ ] CI/CD integration using GitHub Actions.
- [ ] Implementation of monitoring stack (Prometheus & Grafana).
- [ ] Drift simulation and automated retraining.
