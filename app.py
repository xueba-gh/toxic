import streamlit as st
import pickle
import time
from sklearn.feature_extraction.text import TfidfVectorizer

# We'll Load the trained model
model = pickle.load(open('/home/kwamena/Desktop/desk/Fante/random-forest/model.pkl', 'rb'))

#then, Load the TF-IDF vectorizer used during training
tfidf_vectorizer = pickle.load(open('/home/kwamena/Desktop/desk/Fante/random-forest/tfidf_vectorizer.pkl', 'rb'))

# Styling
st.markdown(
    """
    <style>
        .reportview-container {
            background: #00000;
            padding: 1rem;
        }
        .big-font {
            font-size: 40px !important;
        }
        .centered {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 300px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and description
st.markdown("<p class='big-font'> 🔍 Toxic Comment Detector 🔍</p>", unsafe_allow_html=True)
st.write(
    "This application predicts whether a given comment is toxic or not. For training purposes only"
)

# User input on our web app
message = st.text_area('Enter the comment to predict', height=10)
submit = st.button('Predict')

if message and submit:
   with st.spinner('Predicting...'):
        

        # Preprocess the input message using the same TF-IDF vectorizer
        message_tfidf = tfidf_vectorizer.transform([message])

        # Make the prediction
        prediction = model.predict(message_tfidf)

        # Display prediction result
        st.markdown("<div class='centered big-font'>", unsafe_allow_html=True)
        if prediction[0] == 1:
            st.error('🚨 Warning: This comment seems toxic!')
        else:
            st.success('✅ Great! This comment is not toxic.')
        st.markdown("</div>", unsafe_allow_html=True)
elif not message and submit:
    st.warning('Please enter a comment to predict.')


confidence_level = 0.74 
st.markdown(f"**Confidence Level:** {confidence_level * 100:.2f}%")

# Additional information about this model
st.markdown(
    """
    **Model Information:**
    - Trained on toxic comment data.
    - Random Forest Classifier.
    - Sample comments include explicit language.
    - For demonstration purposes only.
    """
)