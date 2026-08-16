
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Project directory containing app.py
BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Personalized Course Recommender",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        margin-bottom: 30px;
    }

    .course-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        margin-bottom: 15px;
    }

    .score {
        font-size: 16px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_courses():

    return pd.read_csv(
        BASE_DIR / "data" / "courses.csv"
    )


@st.cache_data
def load_interactions():

    return pd.read_csv(
        BASE_DIR / "data" / "interactions.csv"
    )


@st.cache_resource
def load_models():

    course_similarity = joblib.load(
        BASE_DIR / "models" / "course_similarity.pkl"
    )

    interaction_matrix = joblib.load(
        BASE_DIR / "models" / "interaction_matrix.pkl"
    )

    nmf_predictions = joblib.load(
        BASE_DIR / "models" / "nmf_predictions.pkl"
    )

    return (
        course_similarity,
        interaction_matrix,
        nmf_predictions
    )


courses = load_courses()
interactions = load_interactions()

(
    course_similarity,
    interaction_matrix,
    nmf_prediction_df
) = load_models()


# ============================================================
# SCORE NORMALIZATION
# ============================================================

def min_max_normalize(scores):

    scores = np.array(
        scores,
        dtype=float
    )

    if len(scores) == 0:
        return scores

    min_score = scores.min()
    max_score = scores.max()

    if max_score == min_score:
        return np.ones_like(scores)

    return (
        (scores - min_score)
        /
        (max_score - min_score)
    )


# ============================================================
# HYBRID RECOMMENDER
# ============================================================

def recommend_courses(
    user_id,
    completed_courses,
    top_n=5,
    content_weight=0.70,
    nmf_weight=0.30
):

    completed_courses = [
        course
        for course in completed_courses
        if course in courses["course_name"].values
    ]

    if len(completed_courses) == 0:
        return pd.DataFrame()

    # --------------------------------------------------------
    # CONTENT-BASED SCORE
    # --------------------------------------------------------

    content_scores = np.zeros(
        len(courses)
    )

    for course_name in completed_courses:

        course_index = courses[
            courses["course_name"] == course_name
        ].index[0]

        content_scores += (
            course_similarity[
                course_index
            ]
        )

    content_scores /= len(
        completed_courses
    )

    # --------------------------------------------------------
    # NMF COLLABORATIVE SCORE
    # --------------------------------------------------------

    nmf_scores = np.zeros(
        len(courses)
    )

    if user_id in nmf_prediction_df.index:

        user_scores = (
            nmf_prediction_df.loc[user_id]
        )

        for course_id, score in user_scores.items():

            course_index = courses[
                courses["course_id"] == course_id
            ].index

            if len(course_index) > 0:

                nmf_scores[
                    course_index[0]
                ] = score

    # --------------------------------------------------------
    # NORMALIZE BOTH SCORES
    # --------------------------------------------------------

    content_normalized = (
        min_max_normalize(
            content_scores
        )
    )

    nmf_normalized = (
        min_max_normalize(
            nmf_scores
        )
    )

    # --------------------------------------------------------
    # HYBRID SCORE
    # --------------------------------------------------------

    final_scores = (
        content_weight * content_normalized
        +
        nmf_weight * nmf_normalized
    )

    # --------------------------------------------------------
    # REMOVE COURSES ALREADY COMPLETED
    # --------------------------------------------------------

    for course_name in completed_courses:

        index = courses[
            courses["course_name"] == course_name
        ].index[0]

        final_scores[index] = -1

    # --------------------------------------------------------
    # SELECT TOP-N COURSES
    # --------------------------------------------------------

    top_indices = np.argsort(
        final_scores
    )[::-1][:top_n]

    recommendations = courses.iloc[
        top_indices
    ][
        [
            "course_id",
            "course_name",
            "genre",
            "description"
        ]
    ].copy()

    recommendations[
        "content_score"
    ] = content_normalized[
        top_indices
    ]

    recommendations[
        "nmf_score"
    ] = nmf_normalized[
        top_indices
    ]

    recommendations[
        "final_score"
    ] = final_scores[
        top_indices
    ]

    return recommendations.reset_index(
        drop=True
    )


# ============================================================
# APPLICATION HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎓 Personalized Course Recommender</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Discover courses personalized to your learning history
    using machine learning and recommender-system techniques.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "👤 Your Learning Profile"
)

st.sidebar.write(
    "Select the courses you have already completed."
)

selected_courses = st.sidebar.multiselect(
    "Completed courses",
    options=courses["course_name"].tolist()
)

top_n = st.sidebar.slider(
    "Number of recommendations",
    min_value=3,
    max_value=10,
    value=5
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if len(selected_courses) == 0:

    st.info(
        "👈 Select at least one completed course "
        "from the sidebar to get recommendations."
    )

    st.subheader(
        "How the system works"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            "### 📚 1. Course Content"
        )

        st.write(
            "Course names, topics and descriptions "
            "are represented using TF-IDF."
        )

    with col2:

        st.markdown(
            "### 🧠 2. Machine Learning"
        )

        st.write(
            "Content similarity and NMF collaborative "
            "filtering identify relevant courses."
        )

    with col3:

        st.markdown(
            "### 🎯 3. Recommendation"
        )

        st.write(
            "The two signals are combined to rank "
            "courses personalized to the learner."
        )

else:

    st.subheader(
        "📚 Your Completed Courses"
    )

    for course in selected_courses:

        st.write(
            f"✓ {course}"
        )

    st.divider()

    # --------------------------------------------------------
    # GENERATE RECOMMENDATIONS
    # --------------------------------------------------------

    recommendations = recommend_courses(
        user_id=1,
        completed_courses=selected_courses,
        top_n=top_n
    )

    st.subheader(
        "🎯 Recommended For You"
    )

    if recommendations.empty:

        st.warning(
            "No recommendations could be generated."
        )

    else:

        for rank, row in recommendations.iterrows():

            score = (
                row["final_score"] * 100
            )

            st.markdown(
                f"""
                <div class="course-card">

                <h3>
                {rank + 1}. {row["course_name"]}
                </h3>

                <p>
                <b>Category:</b>
                {row["genre"]}
                </p>

                <p>
                {row["description"]}
                </p>

                <p class="score">
                Recommendation Score:
                {score:.1f}%
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        st.caption(
            "Final score = 70% content similarity + "
            "30% collaborative preference."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Machine Learning Capstone Project | "
    "Personalized Course Recommender System"
)
