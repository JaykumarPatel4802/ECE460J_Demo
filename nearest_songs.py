import pandas as pd
from sklearn.neighbors import NearestNeighbors

sample_dataset = pd.read_csv('sampled_dataset_final.csv')
small_test_dataset = sample_dataset.sample(n=100)

def knn(new_features, k):
    features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    knn = NearestNeighbors(n_neighbors=k).fit(small_test_dataset[features])

    distances, indices = knn.kneighbors(new_features)
    return indices

def get_songs(indices):
    songs = {}
    for i in indices[0]:
        song_name = sample_dataset.loc[i, 'track_name']
        artist_name = sample_dataset.loc[i, 'artists']
        songs[song_name] = artist_name
    return songs

new_features = [[0.637, 0.643, 4, -6.571, 1, 0.0519, 0.13, 1.8e06, 0.142, 0.533, 97.008]]
k = 5
indices = knn(new_features, k)
songs = get_songs(indices)
print(songs)

# print(sample_dataset.iloc[30])