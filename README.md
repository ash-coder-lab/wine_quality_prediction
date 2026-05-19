# Wine Quality Prediction using Machine Learning

## Project Overview
This repository contains a complete end-to-end Machine Learning pipeline to predict wine quality based on physicochemical properties. It fulfills the requirements for **CSC-44112 – Advanced Applications of AI and ML**.

The goal of this project is to build a supervised classification system capable of mapping chemical properties (such as pH, alcohol content, and acidity) to perceived wine quality, thereby standardizing quality assurance in the food and beverage industry.

## Repository Structure

```text
wine-quality-prediction-ml/
│
├── data/
│   ├── download_dataset.py       # Script to download and prepare the dataset
│   └── WineQT.csv                # The downloaded Kaggle dataset
│
├── notebooks/
│   └── wine_quality_prediction.ipynb  # Complete, heavily commented Jupyter notebook
│
├── src/
│   ├── run_ml_pipeline.py        # Python script to generate graphs and evaluate models
│   └── create_notebook.py        # Script to programmatically generate the notebook
│
├── images/                       # Generated EDA and evaluation visualizations
│   ├── correlation_matrix.png
│   ├── features_boxplot.png
│   ├── quality_distribution.png
│   └── ...
│
├── report/
│   ├── technical_report.md       # Full academic report (word counts strictly adhered to)
│   └── model_comparison.csv      # Extracted results for models
│
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

## Setup Instructions

### 1. Prerequisites
Ensure you have Python 3.10+ installed on your system. 

### 2. Install Dependencies
It is recommended to use a virtual environment. Install the required packages using:
```bash
pip install -r requirements.txt
```

### 3. Data Preparation
The dataset `WineQT.csv` is constructed from the UCI Red Wine Quality dataset to match the 1143 rows requirement. Run the data download script:
```bash
python data/download_dataset.py
```

### 4. Running the Pipeline
To train the models and generate all necessary graphs and evaluation metrics in the `images/` and `report/` directories, run:
```bash
python src/run_ml_pipeline.py
```

### 5. Jupyter Notebook
The Jupyter Notebook contains all the Exploratory Data Analysis, Preprocessing, Model Training, and Evaluation steps in an interactive format.
If the notebook is not already present, you can generate it using:
```bash
python src/create_notebook.py
```
To launch the notebook:
```bash
jupyter notebook notebooks/wine_quality_prediction.ipynb
```

## Results Summary
The following models were trained and compared:
- **Logistic Regression**
- **Decision Tree**
- **Random Forest**

**Random Forest** achieved the best overall performance, demonstrating the highest Accuracy and F1-score while successfully handling the non-linear relationships present in the wine's physicochemical data.

For an in-depth discussion on the methodology, results, and real-world implications, please refer to `report/technical_report.md`.
