# 🎓 Personalized Course Recommender System

A machine-learning based personalized course recommendation system developed as a Machine Learning Capstone Project.

## Overview

This project recommends online courses based on a learner's previous course history.

The system combines content-based filtering and collaborative filtering to generate personalized course recommendations.

## Machine Learning Techniques

- TF-IDF
- Cosine Similarity
- Content-Based Filtering
- K-Nearest Neighbours Collaborative Filtering
- PCA
- KMeans Clustering
- Non-negative Matrix Factorization
- Neural Network Embeddings
- Regression
- Classification
- Hybrid Recommendation

## Final Recommendation Model

The deployed application uses a hybrid recommendation strategy:

**70% Content-Based Similarity + 30% NMF Collaborative Filtering**

Both scores are normalized before being combined.

Completed courses are removed from the final recommendation list.

## Dataset

The course catalogue contains courses related to:

- Python
- Machine Learning
- Deep Learning
- Data Science
- SQL
- Cloud Computing
- Big Data
- Artificial Intelligence
- Computer Vision

The learner interaction and rating data used for collaborative filtering is simulated for experimentation.

## Application

The Streamlit application allows a learner to:

1. Select courses they have already completed.
2. Choose how many recommendations they want.
3. Receive personalized course recommendations.
4. View the recommendation score and course category.

## Project Structure

course_recommender/

├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── courses.csv
│   └── interactions.csv
│
└── models/
    ├── course_similarity.pkl
    ├── interaction_matrix.pkl
    ├── nmf_model.pkl
    ├── nmf_predictions.pkl
    └── tfidf_vectorizer.pkl

## How to Run

Install the dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

## Recommendation Methodology

### Content-Based Filtering

Course names, genres and descriptions are represented using TF-IDF. Cosine similarity is then used to determine how closely related courses are.

### NMF Collaborative Filtering

The learner-course interaction matrix is decomposed into latent factors using Non-negative Matrix Factorization.

### Hybrid Recommendation

The final recommendation score is:

Final Score = 0.70 × Content Similarity + 0.30 × Collaborative Preference

This allows the system to combine course-content relevance with learner preference patterns.

## Limitations

The learner interaction dataset is simulated. Therefore, collaborative filtering recommendations may contain some variability.

A future version could use real learner interactions, course completion history, implicit feedback, timestamps and larger-scale datasets.

## Author

Ayush Parashar