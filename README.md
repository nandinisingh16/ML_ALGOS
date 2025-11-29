
---

#  Machine Learning Algorithms — Complete End-to-End Implementation

This repository contains a **full suite of Machine Learning models**, including:

* **Regression models** (Linear, Ridge, Lasso, Elastic Net)
* **Classification models** (Logistic Regression, Naive Bayes, KNN, SVM, Decision Tree, Ensemble Models)
* **Boosting & Bagging techniques**
* **XGBoost-ready structure**
* **Clustering techniques** (KMeans, Agglomerative Clustering)
* **Anomaly detection** (Isolation Forest, DBSCAN, LOF)
* **Visualization of decision boundaries, residuals & clusters**

The code uses **California Housing**, **Iris**, **Breast Cancer** datasets and synthetic datasets (make_circles, make_blobs).

---

##  Project Structure

The notebook/script includes the following major sections:

###  **1. Regression Models**

* Linear Regression
* Ridge Regression (Hyperparameter tuning with GridSearchCV)
* Lasso Regression
* Elastic Net Regression
* SVM Regression

Includes:

* Residual plots
* R² scores
* Grid search optimization

---

###  **2. Classification Models**

* Logistic Regression (with penalty tuning: L1, L2, ElasticNet)
* Decision Boundary Visualization
* Naive Bayes
* K-Nearest Neighbors (with scaling)
* SVM Classifier
* Decision Tree (Pre-pruning & Post-pruning)

Includes:

* Confusion matrix
* Classification Report
* Accuracy scores

---

###  **3. Ensemble Learning**

* **Bagging Classifier**
* **Random Forest Classifier**
* **AdaBoost Classifier**
* **Gradient Boosting Classifier**

Includes:

* Evaluation metrics
* Comparison across models

---

###  **4. Anomaly Detection**

* Isolation Forest
* DBSCAN
* Local Outlier Factor

Includes:

* Scatter plots
* Highlighting outliers

---

###  **5. Clustering Algorithms**

* K-Means Clustering (with Silhouette Score)
* Agglomerative Hierarchical Clustering
* Dendrogram Plot

Includes:

* Cluster visualizations
* Centroid plotting

---

##  Requirements

```txt
python >= 3.8
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
```


---

##  How to Run

Just execute the cells in the provided Python script / Jupyter Notebook:

```bash
python main.py
```

Or open the notebook:

```bash
jupyter notebook
```

---

##  Visualizations Included

* KDE residual plots
* Decision boundaries
* Cluster visualization
* Outlier detection highlights
* Tree diagrams (Decision Trees)

---


---



---
