import pandas as pd

print("Loading Dataset...")

df = pd.read_csv(
    "data/spam.csv",
    encoding="latin-1"
)

print("Dataset Loaded Successfully!")

print(df.head())

print("\nColumns:")
print(df.columns)

# Keep only useful columns

df = df[['v1', 'v2']]

df.columns = [
    'label',
    'message'
]

print("\nCleaned Dataset:")

print(df.head())

df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

print("\nLabel Conversion Complete!")
print(df.head())

from sklearn.model_selection import train_test_split

X = df['message']

y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain-Test Split Complete!")

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF Vectorizer

vectorizer = TfidfVectorizer()

# Convert text into numerical vectors

X_train = vectorizer.fit_transform(X_train)

X_test = vectorizer.transform(X_test)

print("\nTF-IDF Vectorization Complete!")

print("Training Matrix Shape:", X_train.shape)

print("Testing Matrix Shape:", X_test.shape)

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

model.fit(
    X_train,
    y_train
)

print("\nNaive Bayes Model Trained Successfully!")

predictions = model.predict(X_test)

print("\nFirst 10 Predictions:")

for i in range(10):

    print(
        "Actual:",
        y_test.iloc[i],
        "| Predicted:",
        predictions[i]
    )

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    "\nAccuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)

import pickle

pickle.dump(
    model,
    open(
        "model.pkl",
        "wb"
    )
)

pickle.dump(
    vectorizer,
    open(
        "vectorizer.pkl",
        "wb"
    )
)

print(
    "\nModel Saved Successfully!"
)

print(
    "Vectorizer Saved Successfully!"
)