# Import libraries
import streamlit as st
import pandas as pd
import numpy as np

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from getpass import getpass

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# Load dataframes
billboard = pd.read_csv("billboard_songs.csv")
audio_features_songs = pd.read_csv("audio_features_songs_final.csv")
X = pd.read_csv("X_scaled.csv")

# Start Spotipy with user credentials
client_id = "4617eb96ef9545239b68c0d443d5baba"
client_secret = "93d966e07836444e9f4a9ddd881f0060"
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id, client_secret))

# Train model and predict
kmeans = KMeans(n_clusters=4, n_init=10)
kmeans.fit(X)
clusters = kmeans.predict(X)
songs_clusters = audio_features_songs.copy()
songs_clusters["cluster"] = clusters

# Function to recommend song
def recommend_song(title, artist):
    # If the user input is a song in the Billboard list, recommend another song from the list
    if title in billboard["title"].unique():
        if artist in list(billboard.loc[billboard["title"] == title, "artist"]):
            while True:
                recommendation = billboard.sample(1, ignore_index=True)
                if recommendation["title"][0] != title or recommendation["artist"][0] != artist:
                    return ' - '.join([recommendation["title"][0], recommendation["artist"][0]])
                    break

                    # If it's not in the Billboard list, try to find the song in Spotipy, if not return an error message
    elif sp.search(q=f"track:{title}", limit=1)["tracks"]["total"] == 0:
        return "Could not find the song"

    # When the song is in Spotipy, search for a similar song by clustering
    else:

        # Obtain audio features of the user song
        results = sp.search(q=f"track:{title}", limit=1)
        track_id = results["tracks"]["items"][0]["id"]
        audio_features = sp.audio_features(track_id)

        # Create dataframe
        df = pd.DataFrame(audio_features)
        new_features = df[X.columns]

        # Scale features
        X_scaled = MinMaxScaler().fit_transform(new_features)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        # Predict cluster
        cluster = kmeans.predict(X_scaled)

        # Filter the dataframe to select only the predicted cluster
        filtered_df = songs_clusters[songs_clusters["cluster"] == cluster[0]]

        # Get one random song from the filtered dataframe
        recommendation = filtered_df.sample(1, ignore_index=True)
        return ' - '.join([recommendation["title"][0], recommendation["artist"][0]])

# Defining main function
def main():

    # Title
    st.title(":notes: Song recommender :notes:")

    # Store the initial value of widgets in session state
    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False

    # User input
    user_title = st.text_input(":musical_keyboard: Enter the song title :musical_keyboard:")
    if user_title:
        st.write("You entered: ", user_title)

    user_artist = st.text_input(":dancers: Enter the artist :dancers:")
    if user_artist:
        st.write("You entered: ", user_artist)

    # Create button for running the recommender
    if st.button("RECOMMEND"):
        recommendation = recommend_song(user_title, user_artist)
        st.success(recommendation)

if __name__ == '__main__':
    main()