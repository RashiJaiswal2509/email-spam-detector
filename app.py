import streamlit as st
import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="",
    layout="centered"
)

st.title("Email Spam Detector")
st.write("Detect whether a message is Spam or Not Spam")

message = st.text_area(
    "Enter Message",
    height=150
)

if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:

        transformed = vectorizer.transform([message])

        prediction = model.predict(transformed)[0]

        probability = model.predict_proba(transformed)

        confidence = max(probability[0]) * 100

        if prediction == 1:
            st.error("Spam Message Detected")
        else:
            st.success("Not Spam")

        st.write(
            f"### Confidence Score: {confidence:.2f}%"
        )

        st.progress(int(confidence))

        st.subheader("Prediction Details")

        st.write(
            {
                "Spam Probability":
                    round(probability[0][1] * 100, 2),

                "Not Spam Probability":
                    round(probability[0][0] * 100, 2)
            }
        )