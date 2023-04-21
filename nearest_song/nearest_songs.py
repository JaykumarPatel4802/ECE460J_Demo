import pandas as pd
from sklearn.neighbors import NearestNeighbors

def knn(filtered_dataset, new_features, k):
    features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    knn = NearestNeighbors(n_neighbors=k).fit(filtered_dataset[features])

    distances, indices = knn.kneighbors(new_features)
    return indices

def get_songs(filtered_dataset, indices):
    songs = dict()
    for i in indices[0]:
        # song_name = filtered_dataset.loc[i, 'track_name']
        print(filtered_dataset.iloc[[i]])
        song_name = filtered_dataset.iloc[[i]]['track_name']
        print("song name: ", song_name.values[0])
        song_name = song_name.values[0]
        # artist_name = filtered_dataset.loc[i, 'artists']
        artist_name = filtered_dataset.iloc[[i]]['artists']
        print("artist name: ", artist_name.values[0])
        artist_name = artist_name.values[0]
        songs[song_name] = artist_name
    return songs

def get_nearest_songs(track_ids, features, k):
    print("track_ids: ", track_ids)
    print("features: ", features)
    print("k: ", k)
    # new_features = [[0.637, 0.643, 4, -6.571, 1, 0.0519, 0.13, 1.8e06, 0.142, 0.533, 97.008]]
    # k = 5
    sample_dataset = pd.read_csv('nearest_song/sampled_dataset_final.csv')
    # filter sample_dataset to only include track_ids
    filtered_dataset = sample_dataset[sample_dataset['track_id'].isin(track_ids)]
    print("filtered_dataset: ", filtered_dataset.shape)
    print("filtered_dataset: ", filtered_dataset.head())
    indices = knn(filtered_dataset, features, k)
    print("indices: ", indices)
    songs = get_songs(filtered_dataset, indices)
    return songs

# print(sample_dataset.iloc[30])