# 📺 YouTube Spam Detection with Multinomial Naive Bayes

This project applies Natural Language Processing and Machine Learning techniques to classify YouTube comments as spam or not spam using the Multinomial Naive Bayes algorithm.

---

## 📂 Dataset

The dataset consists of 5 merged files (`Youtube01.csv` to `Youtube05.csv`) containing real YouTube comments with spam labels.

Features include:
- `COMMENT_ID`, `AUTHOR`, `DATE`, `CONTENT`, `CLASS`

---

## 🚀 Objectives

- Clean and preprocess YouTube comments
- Convert text data into numerical features using TF-IDF
- Train and evaluate a **Multinomial Naive Bayes** classifier
- Measure performance using accuracy, confusion matrix, and classification report

---

## 🧠 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Matplotlib (for visualization)
- Jupyter Notebook

---

## 📈 Results

- Achieved high accuracy on test data
- Built a full pipeline for text classification

---

## 🛠️ How to Run

1. Clone the repository  
2. Install dependencies (`pip install -r requirements.txt`)  
3. Open the notebook:  
```bash
jupyter notebook YouTube_Spam_Detection.ipynb

 
