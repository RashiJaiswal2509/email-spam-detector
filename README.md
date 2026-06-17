# Email Spam Classifier System

# Project Overview

This project is a **Machine Learning application** that classifies messages as **Spam or Not Spam** based on their text content.

The system uses Natural Language Processing (NLP) techniques to process text data and a **Naive Bayes classification model** to predict whether a message is spam.

---

# Problem Statement

With the increasing number of spam messages in emails and SMS, it becomes difficult to manually filter them.

---

# Features

- Load and preprocess dataset
- Text cleaning and label conversion
- TF-IDF feature extraction
- Train Machine Learning model
- Predict spam or ham messages
- Evaluate model performance
- Save trained model using Pickle
- Interactive Streamlit web application

---

# Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Streamlit  
- Pickle  
- NLP (TF-IDF Vectorizer)

---

# How to Run the Project

1. Install required libraries

pip install pandas numpy scikit-learn streamlit

2. Train the model

python src/train_model.py

3. Run Streamlit app

streamlit run app.py

---

# Future Improvements

- Deploy on Streamlit Cloud or Render
- Use deep learning models (LSTM / BERT)
- Improve dataset preprocessing
- Add email file upload feature
- Integrate Gmail API

---

#  Conclusion

This project demonstrates the use of Machine Learning and NLP for spam detection. It effectively classifies messages with high accuracy and provides an easy-to-use interface using Streamlit.

# Author

Rashi Jaiswal

AI & ML Internship Minor Project
