import streamlit as st
import pickle
import numpy as np
from scipy.sparse import hstack

# ---------------- Page setup ----------------
st.set_page_config(
    page_title="Job Post Review Tool",
    page_icon="📄",
    layout="centered"
)

# ---------------- Load model ----------------
@st.cache_resource
def load_assets():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    return model, tfidf

model, tfidf = load_assets()

# ---------------- Header ----------------
st.title("Job Post Review Tool")
st.caption("A simple system to flag potentially risky job postings")

st.markdown(
    """
This tool reviews a job description and highlights  
whether it **looks safe or potentially suspicious** based on past patterns.
"""
)

st.divider()

# ---------------- Input ----------------
st.subheader("Job description")

job_text = st.text_area(
    "Paste the job description below",
    height=220,
    placeholder="Example: We are hiring a remote data analyst..."
)

st.subheader("Missing information (if any)")

salary_missing = st.checkbox("Salary range not mentioned")
company_missing = st.checkbox("Company profile not provided")
requirements_missing = st.checkbox("Job requirements missing")
benefits_missing = st.checkbox("Benefits not mentioned")
department_missing = st.checkbox("Department not specified")
employment_missing = st.checkbox("Employment type unclear")

st.divider()

# ---------------- Prediction ----------------
if st.button("Review job posting"):
    if not job_text.strip():
        st.warning("Please enter a job description to continue.")
    else:
        text_vec = tfidf.transform([job_text])

        flags = np.array([[
            salary_missing,
            company_missing,
            requirements_missing,
            benefits_missing,
            department_missing,
            employment_missing
        ]])

        X_input = hstack([text_vec, flags])

        score = model.decision_function(X_input)[0]
        prediction = 1 if score > 0 else 0

        st.divider()

        if prediction == 1:
            st.error("⚠️ This job posting looks risky")

            st.markdown(
                """
                Based on similar job posts seen in the past,  
                this one shares **multiple warning signs**.

                **Why this might need attention:**
                - Important details are missing or vague  
                - The wording resembles known scam patterns  
                - Low transparency overall  

                This does **not** guarantee fraud,  
                but it is worth reviewing carefully.
                """
            )
        else:
            st.success("✅ This job posting looks normal")

            st.markdown(
                """
                The job description appears **clear and structured**,  
                and it does not match common scam patterns.

                **Positive signs:**
                - Information is reasonably complete  
                - Language is consistent with legitimate postings  
                - No strong red flags detected  

                Still, basic verification is always recommended.
                """
            )

# ---------------- Footer ----------------
st.divider()
st.caption(
    "Built as a data science project using text analysis and classification"
)
