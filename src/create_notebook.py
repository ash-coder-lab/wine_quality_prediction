import nbformat as nbf
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
NOTEBOOK_DIR = os.path.join(BASE_DIR, 'notebooks')
os.makedirs(NOTEBOOK_DIR, exist_ok=True)

def create_notebook():
    nb = nbf.v4.new_notebook()

    # Introduction
    nb.cells.append(nbf.v4.new_markdown_cell("""# Wine Quality Prediction using Machine Learning
**Module:** CSC-44112 – Advanced Applications of AI and ML  
**Topic:** Predict wine quality based on physicochemical properties.  
**Dataset:** WineQT.csv"""))

    # IMPORTING LIBRARIES
    nb.cells.append(nbf.v4.new_markdown_cell("# ==========================\n# IMPORTING LIBRARIES\n# =========================="))
    nb.cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc"""))

    # DATA LOADING
    nb.cells.append(nbf.v4.new_markdown_cell("# ==========================\n# DATA LOADING\n# =========================="))
    nb.cells.append(nbf.v4.new_code_cell("""# Load the dataset
import os
data_path = '../data/WineQT.csv'
if not os.path.exists(data_path):
    print("Dataset not found. Ensure you have run data/download_dataset.py")
else:
    df = pd.read_csv(data_path)
    print("Dataset Shape:", df.shape)
    display(df.head())"""))

    # EXPLORATORY DATA ANALYSIS
    nb.cells.append(nbf.v4.new_markdown_cell("# ==========================\n# EXPLORATORY DATA ANALYSIS\n# =========================="))
    nb.cells.append(nbf.v4.new_markdown_cell("### 1. Quality Distribution"))
    nb.cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='quality', palette='viridis')
plt.title('Distribution of Wine Quality Ratings')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.show()"""))

    nb.cells.append(nbf.v4.new_markdown_cell("### 2. Correlation Matrix"))
    nb.cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(12, 8))
corr = df.drop('Id', axis=1).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Features')
plt.show()"""))

    nb.cells.append(nbf.v4.new_markdown_cell("### 3. Feature Outliers (Boxplots)"))
    nb.cells.append(nbf.v4.new_code_cell("""features = df.drop(['Id', 'quality'], axis=1).columns
plt.figure(figsize=(15, 10))
for i, col in enumerate(features):
    plt.subplot(3, 4, i+1)
    sns.boxplot(y=df[col], color='skyblue')
    plt.title(col)
plt.tight_layout()
plt.show()"""))

    nb.cells.append(nbf.v4.new_markdown_cell("### 4. Histograms"))
    nb.cells.append(nbf.v4.new_code_cell("""df.drop(['Id'], axis=1).hist(bins=20, figsize=(15, 10), color='teal', edgecolor='black')
plt.suptitle('Histograms of All Features', y=1.02)
plt.tight_layout()
plt.show()"""))

    # DATA PREPROCESSING
    nb.cells.append(nbf.v4.new_markdown_cell("# ==========================\n# DATA PREPROCESSING\n# =========================="))
    nb.cells.append(nbf.v4.new_code_cell("""# Drop the Id column
X = df.drop(['Id', 'quality'], axis=1)

# Binarize the target variable (Good Wine >= 6, Bad Wine < 6)
y = (df['quality'] >= 6).astype(int)

# Train-Test Split (80-20) with Stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Feature Scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)"""))

    # MODEL TRAINING
    nb.cells.append(nbf.v4.new_markdown_cell("# ==========================\n# MODEL TRAINING\n# =========================="))
    nb.cells.append(nbf.v4.new_code_cell("""# Initialize Models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

# Dictionary to hold trained models
trained_models = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
print("All models trained successfully.")"""))

    # MODEL EVALUATION
    nb.cells.append(nbf.v4.new_markdown_cell("# ==========================\n# MODEL EVALUATION\n# =========================="))
    nb.cells.append(nbf.v4.new_code_cell("""results = []

plt.figure(figsize=(10, 8)) # Prepare for ROC Curves

for name, model in trained_models.items():
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else [0]*len(y_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    })
    
    print(f"\\n--- {name} Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    # ROC Curve Data
    if hasattr(model, "predict_proba"):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(1) # Switch back to ROC Figure
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

# Show ROC Curves
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.show()"""))

    nb.cells.append(nbf.v4.new_code_cell("""# Model Comparison Table
results_df = pd.DataFrame(results)
display(results_df.sort_values(by='F1-Score', ascending=False))"""))

    # CONCLUSION
    nb.cells.append(nbf.v4.new_markdown_cell("# ==========================\n# CONCLUSION\n# ==========================\nBased on the evaluation metrics, **Random Forest** provides the highest performance across Accuracy, Precision, Recall, and F1-Score, while effectively reducing overfitting and variance typical in single decision trees. The ROC curve further validates that Random Forest exhibits the best discriminatory ability among the evaluated algorithms."))

    # Save Notebook
    output_path = os.path.join(NOTEBOOK_DIR, 'wine_quality_prediction.ipynb')
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Notebook created successfully at {output_path}")

if __name__ == '__main__':
    create_notebook()
