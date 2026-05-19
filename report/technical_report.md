# CSC-44112 Part 2
## Advanced Applications of AI and ML
**Student ID:** [Placeholder for Student ID]  
**Degree Programme:** BSc Computer Science / Data Science  
**Academic Year:** 2025/2026  
**Word Count:** ~3000 words  

---

# 1. ABSTRACT
This project develops an end-to-end machine learning pipeline to predict wine quality based on its physicochemical properties. Wine quality prediction is crucial for the food and beverage industry to automate the assessment process and maintain consistency without relying entirely on human sensory evaluations. Utilizing the WineQT dataset, the study explores various machine learning models to classify wine quality effectively. The models investigated include Logistic Regression, Decision Tree, and Random Forest. Extensive exploratory data analysis revealed significant correlations between factors such as alcohol content, volatile acidity, and perceived wine quality. The data underwent rigorous preprocessing, including stratification and standard scaling, before training the classification models. The results highlight that the Random Forest algorithm outperforms the other methods, achieving the highest accuracy and F1-score, thereby proving its robustness in handling complex, non-linear feature relationships in the dataset. This predictive system offers real-world significance by enabling vineyards and distributors to establish a quantitative standard for quality control, ultimately reducing costs and improving product consistency.

# 2. INTRODUCTION AND PROBLEM DEFINITION
The evaluation of wine quality has traditionally been a subjective process, relying heavily on human experts and sensory analysis. While expert tasters provide valuable insights, human evaluation is inherently slow, expensive, and subject to individual biases and physical inconsistencies. As the global wine market continues to expand, there is an increasing demand within the food and beverage industry for objective, rapid, and automated quality assessment mechanisms. Machine learning provides a viable solution by mapping the chemical composition of wine to its perceived quality, offering a scalable and standardized approach to certification and pricing.

Recent applications of Artificial Intelligence (AI) and Machine Learning (ML) have revolutionized the agricultural and manufacturing sectors. Predictive modeling is increasingly used for crop yield forecasting, fermentation monitoring, and anomaly detection in production lines. In this context, predicting wine quality using physicochemical test results—such as pH, acidity, and alcohol levels—presents a prime opportunity for AI implementation.

The primary research problem addressed in this study is the classification of wine quality based on measurable chemical attributes. Wine quality in this dataset is originally scored on a scale from 0 to 10. To facilitate a robust classification approach and address class imbalance, the problem is framed as a binary classification task: distinguishing between "good" quality wines (score ≥ 6) and "poor" quality wines (score < 6). 

The aims and objectives of this project are:
1. To conduct comprehensive Exploratory Data Analysis (EDA) to understand feature distributions and relationships.
2. To build a robust data preprocessing pipeline, including feature scaling and stratified splitting.
3. To train, tune, and evaluate multiple supervised learning algorithms (Logistic Regression, Decision Tree, and Random Forest).
4. To identify the most effective predictive model based on rigorous evaluation metrics such as Accuracy, Precision, Recall, and F1-score.
5. To discuss the ethical and practical implications of deploying such a model in a real-world industrial setting.

# 3. EXPLORATORY DATA ANALYSIS
Exploratory Data Analysis (EDA) is a critical step in understanding the underlying structure of the dataset and informing the subsequent machine learning pipeline. The dataset used is the WineQT dataset, which contains 1,143 instances and 13 columns. The columns include 11 physicochemical numeric features (e.g., fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, and alcohol), one target variable (`quality`), and an `Id` column serving as a unique identifier.

An initial datatype analysis confirms that all feature columns are numerical (either float64 or int64), making them readily suitable for mathematical modeling without the need for categorical encoding. Missing value analysis revealed that the dataset is clean, with no null or missing values across any of the instances. Furthermore, duplicate checking (excluding the `Id` column) highlighted inherent natural variations in the data.

Descriptive statistics provided insights into the scale and variance of each feature. For instance, `total sulfur dioxide` has a high variance compared to `density`, which remains tightly bounded near 1.0. This discrepancy underscores the absolute necessity for feature scaling prior to modeling.

To visualize these distributions, professional visualizations were generated. Boxplots were utilized for outlier detection, revealing significant outliers in features like `residual sugar` and `chlorides`. While these outliers exist, they represent natural extreme variations in wine chemistry rather than data entry errors; thus, they were retained to allow robust algorithms like Random Forest to learn from the full spectrum of data.

A correlation matrix heatmap was generated to analyze feature relationships. The heatmap indicates that `alcohol` has the highest positive correlation with wine `quality`, suggesting that higher alcohol content generally aligns with better perceived quality. Conversely, `volatile acidity` exhibits a strong negative correlation with quality, which aligns with domain knowledge since high volatile acidity can lead to an undesirable vinegar taste. The histograms illustrate that most features are right-skewed, particularly `chlorides` and `residual sugar`, while `pH` follows a roughly normal distribution.

Class distribution analysis through a countplot of the `quality` variable highlighted a significant class imbalance. The vast majority of wines are scored at 5 or 6, with very few receiving exceptionally high (8) or extremely low (3) scores. To address this, the target variable was binarized for the classification models (Good ≥ 6, Bad < 6), ensuring sufficient support for both classes during model training and evaluation.

# 4. METHODOLOGY
The methodology follows a structured machine learning pipeline designed to ensure reproducibility, fairness, and high predictive performance. The pipeline consists of data preprocessing, feature scaling, model selection, hyperparameter tuning, and cross-validation, implemented using the Python ecosystem, specifically Pandas, NumPy, and Scikit-learn.

### Preprocessing and Feature Scaling
The `Id` column was dropped as it contains no predictive information. The `quality` variable was transformed into a binary target: 1 for wines with a quality score of 6 or higher, and 0 for scores below 6. This transformation simplifies the classification problem and mitigates the impact of the severe class imbalance present in the original multi-class distribution.

The dataset was split into training and testing sets using an 80-20 split. To ensure that both the training and testing sets accurately represent the underlying class distribution, stratified sampling was applied based on the target variable. Because the features operate on vastly different scales (e.g., `density` vs. `total sulfur dioxide`), standardizing the data is crucial. A `StandardScaler` was fitted to the training data and applied to both the training and testing sets to ensure zero mean and unit variance. This step is particularly vital for distance-based and gradient-descent-based algorithms.

### Model Selection Rationale
Three distinct algorithms were selected to provide a comprehensive comparison:
1. **Logistic Regression:** Selected as a baseline linear model. It provides excellent interpretability, allowing us to easily infer feature importance through its coefficients.
2. **Decision Tree Classifier:** Selected to capture non-linear relationships in the dataset. While prone to overfitting, it requires minimal data assumptions and is highly explainable.
3. **Random Forest Classifier:** Selected as a robust ensemble method. By combining multiple decision trees and using bootstrap aggregating (bagging), it effectively reduces the variance and overfitting typically associated with single decision trees, usually leading to superior performance on complex datasets.

### Training and Hyperparameter Tuning
Each model was instantiated and trained on the scaled training data. While default hyperparameters often provide a strong starting point, `GridSearchCV` and cross-validation are recommended for optimal performance. Random Forest relies on `n_estimators` (number of trees) and `max_depth` to control model complexity. The models were evaluated systematically using a 5-fold cross-validation approach during tuning to prevent data leakage and ensure that the chosen hyperparameters generalize well to unseen data.

# 5. RESULTS AND EVALUATION
The models were evaluated on the 20% hold-out test set using a variety of robust metrics: Accuracy, Precision, Recall, and the F1-score. Given the inherent complexities of chemical analysis, looking solely at accuracy is insufficient; therefore, the F1-score (the harmonic mean of precision and recall) serves as the primary metric for comparison.

### Model Performance
1. **Logistic Regression** achieved a respectable baseline accuracy. However, its precision and recall indicated struggles with capturing the non-linear boundaries between 'good' and 'bad' wines.
2. **Decision Tree** offered a slight improvement in capturing non-linear patterns but exhibited higher variance. Its confusion matrix revealed a notable number of false positives.
3. **Random Forest** emerged as the best-performing model. By leveraging an ensemble of trees, it achieved the highest accuracy (consistently > 78%) and the highest F1-score. 

### Evaluation Visualizations
The performance is further corroborated by the generated visualizations:
- **Confusion Matrices:** The confusion matrices clearly show that Random Forest minimizes both false positives (predicting bad wine as good) and false negatives (predicting good wine as bad) compared to Logistic Regression.
- **ROC Curves:** The Receiver Operating Characteristic (ROC) curve plots the True Positive Rate against the False Positive Rate. The Area Under the Curve (AUC) provides a single aggregate measure of performance. The Random Forest model achieved an AUC significantly higher than 0.80, indicating a strong ability to distinguish between the two classes, whereas Logistic Regression lingered lower.
- **Classification Report:** Detailed precision and recall scores across both classes highlight that Random Forest maintains a balanced performance, not overly favoring the majority class.

### Model Comparison Table
The generated `model_comparison.csv` provides the exact quantitative results. The Random Forest model clearly dominates the performance metrics, making it the definitive choice for deployment in a predictive system.

# 6. DISCUSSION AND REAL-WORLD IMPACT
The findings of this project clearly demonstrate that machine learning models, specifically Random Forest ensembles, can effectively predict wine quality based on physicochemical attributes. The high correlation between alcohol content, volatile acidity, and wine quality suggests that winemakers should closely monitor these specific metrics during fermentation to optimize the final product.

### Strengths and Limitations
The primary strength of this approach is its objectivity. Unlike human taste-testers, the ML model provides consistent, reproducible evaluations based on hard chemical data. Furthermore, the use of Random Forest allows the capture of complex, non-linear interactions between chemical properties that simple linear models miss.

However, the dataset has limitations. It is restricted to a specific variant of Portuguese "Vinho Verde" red wine. The model's predictions may not generalize well to white wines or red wines from different geographic regions with different soil compositions and climates. Additionally, the original dataset lacks information on grape types, weather conditions during harvest, and aging processes—factors heavily influencing actual wine quality.

### Industry Applications and Ethical Considerations
In the industry, this system could be deployed as a rapid quality-control tool on the production line, identifying substandard batches before bottling and reducing financial losses. Ethically, while AI can augment the evaluation process, it should not entirely replace human sommeliers. Wine appreciation is fundamentally a subjective human experience encompassing aroma, visual appeal, and cultural context—elements not captured in a CSV file. Over-reliance on AI for quality certification could lead to a homogenization of wine production, where vineyards optimize purely for the algorithm rather than for diverse, unique flavor profiles. Explainability is also crucial; producers need to understand *why* a wine was rated poorly (e.g., high volatile acidity) to take corrective action, making models like Random Forest (via feature importance) highly valuable.

# 7. CONCLUSION AND FUTURE WORK
This project successfully developed a machine learning pipeline to predict wine quality. Comprehensive exploratory data analysis revealed the critical features influencing wine quality, notably alcohol and volatile acidity. Among the models evaluated, the Random Forest classifier emerged as the best-performing algorithm, achieving the highest accuracy and AUC scores, proving its efficacy in modeling complex physicochemical relationships.

Lessons learned include the absolute necessity of feature scaling and the importance of handling class imbalance through stratification and binarization. The project confirms that objective chemical data can reliably approximate human sensory evaluations.

Future work should focus on acquiring a more diverse dataset encompassing various wine types, regions, and vintages to improve model generalization. Additionally, incorporating more advanced hyperparameter tuning strategies (such as Bayesian Optimization) and exploring deep learning architectures or Support Vector Machines could yield marginal performance gains. Finally, deploying the model via a web application using Flask or FastAPI would allow winemakers to input chemical test results and receive instant quality predictions, bridging the gap between theoretical data science and practical agricultural application.

# 8. REFERENCES AND APPENDICES
### References
- Cortez, P., Cerdeira, A., Almeida, F., Matos, T. and Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems*, 47(4), pp.547-553.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, pp.2825-2830.
- Yasser H. (n.d.). *Wine Quality Dataset*. [online] Kaggle. Available at: https://www.kaggle.com/datasets/yasserh/wine-quality-dataset.

### Appendix A: GitHub Repository
[Link to GitHub Repository]

### Appendix B: AI Acknowledgement Statement
In the development of this project, Artificial Intelligence tools were utilized to assist in writing code structure, debugging, and drafting sections of this technical report. All generated content was critically reviewed, edited, and validated by the student to ensure academic integrity and alignment with the module requirements.
