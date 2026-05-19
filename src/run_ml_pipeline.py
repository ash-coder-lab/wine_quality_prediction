import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Don't display plots, just save them
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, roc_auc_score)

# Define directories
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
REPORT_DIR = os.path.join(BASE_DIR, 'report')

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

def perform_eda(df):
    print("Performing EDA...")
    
    # 1. Class Distribution (Quality)
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='quality', palette='viridis')
    plt.title('Distribution of Wine Quality Ratings')
    plt.xlabel('Quality')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'quality_distribution.png'))
    plt.close()

    # 2. Correlation Matrix
    plt.figure(figsize=(12, 8))
    corr = df.drop('Id', axis=1).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix of Features')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'correlation_matrix.png'))
    plt.close()

    # 3. Boxplots (Outliers)
    features = df.drop(['Id', 'quality'], axis=1).columns
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(features):
        plt.subplot(3, 4, i+1)
        sns.boxplot(y=df[col], color='skyblue')
        plt.title(col)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'features_boxplot.png'))
    plt.close()

    # 4. Histograms
    df.drop(['Id'], axis=1).hist(bins=20, figsize=(15, 10), color='teal', edgecolor='black')
    plt.suptitle('Histograms of All Features', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'features_histogram.png'))
    plt.close()

def main():
    data_path = os.path.join(DATA_DIR, 'WineQT.csv')
    if not os.path.exists(data_path):
        print("Dataset not found. Please run download_dataset.py first.")
        return
        
    df = pd.read_csv(data_path)
    print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
    
    perform_eda(df)
    
    print("Preprocessing data...")
    X = df.drop(['Id', 'quality'], axis=1)
    # Convert quality to binary classification (e.g., > 5 is good (1), else bad (0))
    # Wait, the prompt says "predict wine quality based on physicochemical properties"
    # Usually this is binary classification (good vs bad) or multi-class. 
    # Let's make it binary to properly plot ROC curves easily, or just use multi-class ROC.
    # The prompt mentions "ROC curve" which is simpler with binary. Let's create a 'is_good' feature.
    # Actually, standard WineQT task is predicting the exact quality class or binary.
    # Let's do binary classification: 1 if quality >= 6 else 0.
    y = (df['quality'] >= 6).astype(int)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }
    
    results = []
    
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
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
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, f'confusion_matrix_{name.replace(" ", "_")}.png'))
        plt.close()
        
        # ROC Curve (plot all on one graph later, but collect data now)
        if hasattr(model, "predict_proba"):
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.figure(1) # Go back to the ROC figure
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    # Finalize ROC Curve
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'roc_curves_all.png'))
    plt.close()
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(REPORT_DIR, 'model_comparison.csv'), index=False)
    print("Pipeline completed successfully.")
    print(results_df)

if __name__ == "__main__":
    main()
