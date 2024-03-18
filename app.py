import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Spotify API credentials
client_id = 'b0a7268767934c5aa019ba1322c6ccfe'
client_secret = '1f10930334f743a1b9da15dbc37fabbf'
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Read data
df = pd.read_csv("dataset.csv")
df.drop(columns='Unnamed: 0', inplace=True)

dfYear = pd.read_csv("data.csv")
dfYear = dfYear[['id', 'year']]
dfYear['track_id'] = dfYear['id']
dfYear.drop(columns='id', inplace=True)

df = pd.merge(df, dfYear, on='track_id')

# Preprocessing
xtab_song = pd.crosstab(df['track_id'], df['track_genre']) * 2
dfDistinct = df.drop_duplicates('track_id').sort_values('track_id').reset_index(drop=True)
xtab_song.reset_index(inplace=True)
data_encoded = pd.concat([dfDistinct, xtab_song], axis=1)

# Feature scaling
numerical_features = ['explicit', 'danceability', 'energy', 'loudness', 'speechiness', 
                      'acousticness', 'instrumentalness', 'liveness', 'valence', 'year']
scaler = MinMaxScaler()
data_encoded[numerical_features] = scaler.fit_transform(data_encoded[numerical_features])

# Similarity calculation
calculatied_features = numerical_features + list(xtab_song.drop(columns='track_id').columns)
cosine_sim = cosine_similarity(data_encoded[calculatied_features], data_encoded[calculatied_features])

# Streamlit app UI
st.set_page_config(page_title="Music Recommendation App", page_icon="🎵")

st.title("Music Recommendation App")

# Input song title
song_title = st.text_input("Enter a song title:", "Time")

# Number of recommendations
num_recommendations = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)

# Function to get recommendations
def get_recommendations(title, N=5):
    indices = pd.Series(data_encoded.index, index=data_encoded['track_name']).drop_duplicates()
    try:
        idx = indices[title]
        if isinstance(idx, pd.Series):
            idx = idx[0]
    except KeyError:
        return "Song not found in the dataset."

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:N+1]
    song_indices = [i[0] for i in sim_scores]
    recommended_songs = data_encoded[['track_name', 'artists', 'album_name']].iloc[song_indices]

    sim_scores_list = [i[1] for i in sim_scores]
    recommended_list = recommended_songs.to_dict(orient='records')
    for i, song in enumerate(recommended_list):
        song['similarity_score'] = sim_scores_list[i]

    return recommended_list



# Display recommendations
if st.button("Get Recommendations"):
    recommended_songs = get_recommendations(song_title, num_recommendations)
    if isinstance(recommended_songs, str):
        st.write(recommended_songs)
    else:
        st.write("Recommended Songs:")
        for song in recommended_songs:
            st.write(f"Title: {song['track_name']}")
            st.write(f"Artist: {song['artists']}")
            st.write(f"Album: {song['album_name']}")
            st.write(f"Similarity Score: {song['similarity_score']:.2f}")
            st.write("---")
            # Fetch and display album cover
            try:
                track_info = sp.search(q=f"track:{song['track_name']} artist:{song['artists']}", type='track', limit=1)
                if len(track_info['tracks']['items']) > 0:
                    album_cover_url = track_info['tracks']['items'][0]['album']['images'][0]['url']
                    st.image(album_cover_url, caption="Album Cover", width=200)
            except:
                st.write("Album cover not found.")
            # Provide audio preview
            try:
                track_preview_url = track_info['tracks']['items'][0]['preview_url']
                st.audio(track_preview_url, format='audio/mp3', start_time=30)
            except:
                st.write("Audio preview not available.")