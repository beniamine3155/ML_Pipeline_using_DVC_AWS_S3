# ML Pipeline using DVC & AWS S3

This project demonstrates a modular machine learning pipeline for spam detection, managed with DVC and storing artifacts in AWS S3. The pipeline covers data ingestion, preprocessing, feature engineering, model building, and evaluation.

## About the ML Pipeline

The pipeline is designed for reproducible machine learning workflows. Each stage is defined in the `dvc.yaml` file and can be executed independently or as a chain. The main stages are:

- **Data Ingestion**: Download and split raw data.
- **Data Preprocessing**: Clean and format the data.
- **Feature Engineering**: Transform text data using TF-IDF.
- **Model Building**: Train a RandomForestClassifier.
- **Model Evaluation**: Evaluate and log metrics.

See `dvc.yaml` for detailed stage definitions and dependencies.

## DVC Setup & Usage

1. **Install DVC**
	```bash
	pip install dvc
	```
2. **Initialize DVC in your repo**
	```bash
	dvc init
	```
3. **Run the pipeline**
	```bash
	dvc repro
	```
4. **Visualize pipeline DAG**
	```bash
	dvc dag
	```

## Experiment Tracking

1. **Install dvclive**
	```bash
	pip install dvclive
	```
2. **Run experiments**
	```bash
	dvc exp run
	```
3. **Show experiment results**
	```bash
	dvc exp show
	```
4. **VS Code DVC Extension**
	- Install the DVC extension from the VS Code marketplace for integrated experiment tracking.
5. **Remove an experiment**
	```bash
	dvc exp remove <name>
	```
6. **Apply best experiment result**
	```bash
	dvc exp apply <name>
	```

## Adding a Remote S3 Storage to DVC

1. **Create an IAM user and access key** in AWS.
2. **Create an S3 bucket** for storage.
3. **Install required packages**
	```bash
	pip install dvc[s3] awscli boto3
	```
4. **Configure AWS credentials**
	```bash
	aws configure
	# Enter your access key, secret key, and region
	```
5. **Add S3 remote to DVC**
	```bash
	dvc remote add -d dvcstore s3://<bucket_name>
	```
6. **Commit DVC changes**
	```bash
	dvc commit
	```
7. **Push data and models to S3**
	```bash
	dvc push
	```

## Project Structure

## Project Structure

- `data/`: Contains raw, interim, and processed datasets
- `src/`: Source code for each pipeline stage
- `models/`: Trained model artifacts
- `reports/`: Evaluation metrics
- `logs/`: Log files for each stage
- `experiment/`: Jupyter notebook and sample data
- `dvc.yaml`, `params.yaml`: DVC pipeline and parameters

## Pipeline Stages

1. **Data Ingestion** (`src/data_ingestion.py`)
	- Downloads and splits the spam dataset
	- Output: `data/raw/`, `data/interim/`
2. **Data Preprocessing** (`src/data_preprocessing.py`)
	- Cleans and formats the data
	- Output: `data/interim/`
3. **Feature Engineering** (`src/feature_engineering.py`)
	- Applies TF-IDF vectorization
	- Output: `data/processed/`
4. **Model Building** (`src/model_building.py`)
	- Trains a RandomForestClassifier
	- Output: `models/model.pkl`
5. **Model Evaluation** (`src/model_evaluation.py`)
	- Evaluates the model and saves metrics
	- Output: `reports/metrics.json`

## DVC & AWS S3

- DVC tracks data, models, and metrics.
- Artifacts are pushed/pulled from an AWS S3 remote (see `.dvc/config`).

## Setup & Installation

1. Clone the repository:
	```bash
	git clone <repo-url>
	cd ML_Pipeline_using_DVC_AWS_S3
	```
2. Install dependencies:
	```bash
	pip install -r requirements.txt
	```
3. Configure AWS credentials for S3 access (e.g., using `aws configure`).

## Usage

Run the pipeline:
```bash
dvc repro
```

Push/pull data and models to/from S3:
```bash
dvc push   # Uploads tracked files to S3
dvc pull   # Downloads tracked files from S3
```

## Parameters

Pipeline parameters are set in `params.yaml`:
```yaml
data_ingestion:
  test_size: 0.15
feature_engineering:
  max_features: 45
model_building:
  n_estimators: 20
  random_state: 2
```

## Reproducibility & Experiment Tracking

- All pipeline steps are versioned and reproducible with DVC.
- Metrics are logged with dvclive and stored in `reports/metrics.json`.
