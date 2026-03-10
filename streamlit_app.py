import streamlit as st
import joblib
import pandas as pd
import numpy as np
import pickle as pkl

# Load saved objects
df = joblib.load("models/df.pkl")
combined_features = joblib.load("models/combined_features.pkl")
knn = joblib.load("models/knn_model.pkl")

user_course_matrix = joblib.load("models/user_course_matrix.pkl")
latent_matrix = joblib.load("models/latent_matrix.pkl")

st.set_page_config(page_title="Hybrid Course Recommender", layout="centered")
st.title("Online Course Recommendation System")
st.write("Hybrid (Content-Based + Collaborative Filtering)")

# Inputs
user_id = st.number_input("Enter User ID", min_value=1, step=1)
course_id = st.number_input("Enter Course ID", min_value=1, step=1)
top_n = st.slider("Number of recommendations", 1, 10, 5)

# ---------------- Functions ----------------

def content_based_recommend(course_id, top_n=5):
    idx = df[df["course_id"] == course_id].index[0]
    distances, indices = knn.kneighbors(combined_features[idx])

    recs = (
        df.iloc[indices[0][1:]]
        .sort_values("rating", ascending=False)
        .drop_duplicates(subset="course_name")
        .head(top_n)
    )
    return recs

def collaborative_recommend(user_id, top_n=5):
    if user_id not in user_course_matrix.index:
        return None

    user_idx = user_course_matrix.index.get_loc(user_id)
    user_vector = latent_matrix[user_idx]

    scores = latent_matrix @ user_vector
    similar_users = (
        pd.Series(scores, index=user_course_matrix.index)
        .sort_values(ascending=False)
        .iloc[1:6]
    )

    recs = (
        df[df["user_id"].isin(similar_users.index)]
        .sort_values("rating", ascending=False)
        .drop_duplicates(subset="course_name")
        .head(top_n)
    )

    return recs

def hybrid_recommend(user_id, course_id, top_n=5):
    cb = content_based_recommend(course_id, top_n)
    cf = collaborative_recommend(user_id, top_n)

    if cf is None:
        return cb.head(top_n)

    hybrid = (
        pd.concat([cb, cf])
        .drop_duplicates(subset="course_name")
        .head(top_n)
    )

    return hybrid

# ---------------- UI ----------------

if st.button("Recommend Courses"):
    try:
        result = hybrid_recommend(user_id, course_id, top_n)

        # Reset index and start from 1
        result = result.reset_index(drop=True)
        result.index = result.index + 1

        st.subheader("Recommended Courses")
        st.dataframe(result)

    except Exception:
        st.error("Invalid User ID or Course ID")
