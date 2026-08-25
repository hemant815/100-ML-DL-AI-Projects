import joblib
import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.title('Resume Screening')

tfidf = joblib.load(
    "models/tfidf.pkl"
)

classifier = joblib.load(
    "models/Logistic.pkl"
)
@st.cache_resource
def load_model():
    model = SentenceTransformer('all-mpnet-base-v2')
    return model

model = load_model()

def extract_pdf_text(file):
    pdf_byte = file.read()
    doc = fitz.open(stream=pdf_byte, filetype="pdf")

    text = ""

    for page in doc:
        text += page.get_text()

    return text

col1, col2 = st.columns(2)
with col1:
    resume = st.file_uploader('upload_pdf file only',type=['pdf'])

with col2:
    jd = st.text_area('paste your jb')


button = st.button('Score')

if button:
    if resume and jd:
        resume_text = extract_pdf_text(resume)  
        resume_embedding = model.encode(
                [resume_text]
            )

        jd_embedding = model.encode(
                [jd]
            )
        similarity = cosine_similarity(resume_embedding, jd_embedding)[0][0]
        semantic_score = similarity * 100

        st.write(semantic_score)
                # -------------------------
        # TF-IDF Classification
        # -------------------------

        resume_tfidf = tfidf.transform([resume_text])

        prediction = classifier.predict(resume_tfidf)


        # -------------------------
        # Display Results
        # -------------------------

        st.subheader("Results")

        st.write(
            f"Semantic Similarity Score: "
            f"{semantic_score:.2f}%"
        )

        st.write(
            f"Predicted Category: {prediction[0]}"
        )

    else:
        st.warning(
            "Please upload a resume and paste the job description."
        )

        
