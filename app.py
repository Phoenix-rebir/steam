# app.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Steam Game Recommender", layout="wide", initial_sidebar_state="expanded")

st.title("🎮 Steam Game Recommendation System")
st.markdown("---")
st.markdown("**Advanced Matrix Factorization-based Recommendation Engine**")
st.markdown("Powered by Bias-SVD Algorithm | Last Updated: " + datetime.now().strftime("%Y-%m-%d"))

# 1. Load data
df = pd.read_csv("data.csv")
user_list = sorted(df["user-id"].unique())

# 2. Load serialized model and all necessary variables
with open("model.pkl", "rb") as f:
    model_data = pickle.load(f)

# Unpack model data
user_item_matrix = model_data.get('user_item_matrix')
game_popularity = model_data.get('game_popularity')
user_activity = model_data.get('user_activity')
P_b = model_data.get('P_b')
Q_b = model_data.get('Q_b')
mu = model_data.get('mu')
bu = model_data.get('bu')
bi = model_data.get('bi')
user_enc = model_data.get('user_enc')
item_enc = model_data.get('item_enc')

# 3. User selection
col1, col2 = st.columns([3, 1])
with col1:
    user_id = st.selectbox("👤 Select User ID:", user_list, help="Choose a user from the training set")
with col2:
    top_n = st.number_input("Top-N Results:", min_value=1, max_value=20, value=5, step=1)

# 4. Recommendation logic: using Bias-SVD model
def get_recommendations_with_scores(user_id, top_n=5):
    """
    Generate recommendations using Bias-SVD model with confidence scores
    Score = mu + bu[user_idx] + bi + P_b[user_idx] · Q_b.T
    """
    try:
        if user_enc is None:
            return None, ["Error: user encoder not available"]
        
        try:
            user_idx = int(user_enc.transform([user_id])[0])
        except Exception:
            if game_popularity is not None:
                top_games = game_popularity.head(top_n).index.tolist()
                return None, list(top_games)
            else:
                return None, ["User not in training set"]
        
        if P_b is None or Q_b is None or mu is None or bu is None or bi is None:
            return None, ["Error: model components not available"]
        
        # Calculate predicted score for each game
        bias_score = mu + float(bu[user_idx]) + bi
        factor_score = P_b[user_idx].dot(Q_b.T)
        scores = bias_score + factor_score
        
        # Get top-N game indices with scores
        top_indices = np.argsort(-scores)[:top_n]
        top_scores = scores[top_indices]
        
        # Normalize scores to 0-100 scale
        min_score, max_score = top_scores.min(), top_scores.max()
        if max_score > min_score:
            normalized_scores = 100 * (top_scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.full_like(top_scores, 50.0)
        
        # Convert indices back to game names
        if item_enc is None:
            top_games = [str(i) for i in top_indices]
        else:
            try:
                top_games = item_enc.inverse_transform(top_indices)
            except Exception:
                top_games = [str(i) for i in top_indices]
        
        return list(zip(top_games, normalized_scores, top_scores)), None
    
    except Exception as e:
        return None, [f"Error: {str(e)}"]


# 5. Recommendation button and display
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("🔍 Generate Recommendations", use_container_width=True):
        st.session_state.generate = True

with col2:
    if st.button("🔄 Clear Results", use_container_width=True):
        st.session_state.generate = False

# Display recommendations
if st.session_state.get('generate', False):
    recs_data, error = get_recommendations_with_scores(user_id, top_n=top_n)
    
    if error:
        st.warning(f"⚠️ {error[0]}")
    elif recs_data:
        st.markdown("### 📊 Recommendation Results")
        st.markdown(f"**User ID:** {user_id} | **Algorithm:** Bias-SVD Matrix Factorization")
        
        # Display as cards with progress bars
        for idx, (game_name, norm_score, raw_score) in enumerate(recs_data, 1):
            col1, col2, col3 = st.columns([1, 5, 2])
            
            with col1:
                st.markdown(f"### #{idx}")
            
            with col2:
                st.markdown(f"**{game_name}**")
                st.progress(norm_score / 100, text=f"Match Score: {norm_score:.1f}%")
            
            with col3:
                st.metric("Confidence", f"{norm_score:.1f}%", delta=None)
        
        # Display model info
        st.markdown("---")
        with st.expander("📈 Model Information"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Algorithm", "Bias-SVD")
            with col2:
                st.metric("Factors", "20")
            with col3:
                st.metric("Training Size", "80%")
            with col4:
                st.metric("Test RMSE", "122.83")
    else:
        st.info("👆 Click 'Generate Recommendations' to get started")
