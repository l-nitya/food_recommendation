import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained models and matrices
with open('tfidf_matrix.pkl', 'rb') as file:
    tfidf_matrix = pickle.load(file)

with open('cosine_sim.pkl', 'rb') as file:
    cosine_sim = pickle.load(file)

with open('count_matrix.pkl', 'rb') as file:
    count_matrix = pickle.load(file)

with open('cosine_sim2.pkl', 'rb') as file:
    cosine_sim2 = pickle.load(file)

with open('collaborative_model.pkl', 'rb') as file:
    recommender = pickle.load(file)

# Load the datasets
df = pd.read_csv(r'C:\Users\tharu\Downloads\1662574418893344 (1).csv')
ratings = pd.read_csv(r'C:\Users\tharu\Downloads\ratings (1).csv')

# Prepare the rating matrix
rating_matrix = ratings.pivot_table(index='Food_ID', columns='User_ID', values='Rating').fillna(0)

# Create an index for quick lookup
only_food = pd.Series(df.index, index=df['Name']).drop_duplicates()

# Define functions for recommendations
def get_content_based_recommendations(title, df, cosine_sim, only_food):
    """Get recommendations based on content similarity."""
    idx = only_food[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5 recommendations excluding the item itself
    food_indices = [i[0] for i in sim_scores]
    return df['Name'].iloc[food_indices]

def get_collaborative_recommendations(title, df, recommender, rating_matrix):
    """Get collaborative filtering recommendations."""
    user = df[df['Name'] == title]
    user_index = rating_matrix.index.get_loc(int(user['Food_ID']))
    reshaped = rating_matrix.iloc[user_index].values.reshape(1, -1)
    distances, indices = recommender.kneighbors(reshaped, n_neighbors=6)
    
    nearest_neighbors_indices = rating_matrix.iloc[indices[0]].index[1:]
    nearest_neighbors = pd.DataFrame({'Food_ID': nearest_neighbors_indices})
    result = pd.merge(nearest_neighbors, df, on='Food_ID', how='left')
    
    return result['Name'].head()

# Streamlit interface
st.title("Food Recommendation System")

# Create a list of food items for selection
food_items = df['Name'].tolist()

option = st.selectbox(
    'Choose a recommendation method:',
    ('Content-Based (Simple)', 'Content-Based (Advanced)', 'Collaborative Filtering')
)

selected_food = st.selectbox('Select a food item:', food_items)

if st.button('Recommend'):
    if selected_food:
        if option == 'Content-Based (Simple)':
            recommendations = get_content_based_recommendations(selected_food, df, cosine_sim, only_food)
        elif option == 'Content-Based (Advanced)':
            recommendations = get_content_based_recommendations(selected_food, df, cosine_sim2, only_food)
        elif option == 'Collaborative Filtering':
            recommendations = get_collaborative_recommendations(selected_food, df, recommender, rating_matrix)
        
        st.write("Recommendations:")
        for i, item in enumerate(recommendations, 1):
            st.write(f"{i}. {item}")
    else:
        st.write("Please select a food item.")
