# 📧 Spam Detection using TF-IDF & Machine Learning

A simple yet effective machine learning project to classify SMS messages as  **Spam (1)**  or  **Ham (0)**  using  **TF-IDF vectorization**  and two ML models —  **Logistic Regression**  and  **Decision Tree Classifier**.

----------

## 🔍 Project Overview

This project applies Natural Language Processing (NLP) techniques to classify SMS messages.  
The workflow includes:

-   Loading and cleaning the dataset
    
-   Converting text to numerical features using  **TF-IDF**
    
-   Training:
    
    -   **Logistic Regression**
        
    -   **Decision Tree Classifier**
        
-   Evaluating accuracy & confusion matrices
    
-   Visualizing results with Matplotlib & Seaborn
    

----------

## 📁 Dataset

The project uses the  **SMS Spam Collection Dataset**  from Kaggle/UCI.

After loading, only the following columns are used:

Original Column

Renamed As

`v1`

`label`

`v2`

`message`

Labels are converted to numeric:

-   **ham → 0**
    
-   **spam → 1**
    

----------

## 🧰 Technologies Used

-   Python
    
-   Pandas
    
-   NumPy
    
-   Scikit-learn
    
-   Matplotlib
    
-   Seaborn
    

----------

## 🧪 Machine Learning Models

Two models were used:

### **1. Logistic Regression**

-   Well suited for text classification
    
-   Works well with TF-IDF
    
-   Fast & accurate
    

### **2. Decision Tree Classifier**

-   Handles non-linear patterns
    
-   Easy interpretation
    
-   Slightly less accurate for sparse text vectors
    

----------

## 🧼 Preprocessing Steps

1.  Load dataset
    
2.  Select required columns
    
3.  Rename columns
    
4.  Convert labels to numeric
    
5.  Train-test split
    
6.  TF-IDF vectorization (3000 features, English stopwords removed)
    

----------

## 🧮 Model Evaluation

Both models are evaluated using:

-   **Accuracy Score**
    
-   **Confusion Matrix**
    
-   **Classification Report (Precision, Recall, F1-score)**
    
-   **Graphical heatmaps**
    
-   **Bar chart comparison of accuracies**
    

----------

## 📊 Visualizations

### Confusion Matrices

-   Green heatmap → Logistic Regression
    
-   Orange heatmap → Decision Tree
    

### Accuracy Bar Chart

Compares the performance of both models side-by-side.

----------

## 📦 How to Run the Project

`# Clone the repository git clone https://github.com/USERNAME/REPO-NAME.git cd REPO-NAME # Install dependencies pip install -r requirements.txt # Run the script python spam_detector.py` 

> Make sure to update the dataset path according to your system.

----------

## 📝 Code Snippet (Core Pipeline)

`vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_tfidf, y_train)

tree_model = DecisionTreeClassifier(max_depth=20, random_state=42)
tree_model.fit(X_train_tfidf, y_train)` 

----------

## ✔️ Results Summary

Based on typical performance:

Model

Expected Accuracy

Logistic Regression

~95%+

Decision Tree

~88–92%

(Your exact values will come from the output logs.)

----------

## 📌 Future Improvements

-   Add Naive Bayes (great for text classification)
    
-   Add hyperparameter tuning
    
-   Add more preprocessing (stemming, lemmatization)
    
-   Deploy using Flask/Streamlit
    

----------

## 📄 License

This project is licensed under the  **MIT License**.
