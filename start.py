
from requests import request
from sklearn.preprocessing import StandardScaler
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import spotipy.util as util
import json
import numpy as np
import pandas as pd
import torch
import time
import requests

client_id = 'a054865bcfda4b029dad65f46a058b19'
client_secret = os.environ['SPOTIPY_CLIENT_SECRET']

scope = "user-library-read"
username = 'theswifter01'
redirect_uri = 'http://example.com'

# oken = util.prompt_for_user_token(
#   username, scope, client_id, client_secret, redirect_uri)

spotify = spotipy.Spotify(auth_manager=SpotifyOAuth(
    scope=scope, client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri))
user = spotify.current_user()

print("Getting all songs...")


# print(json.dumps(user, sort_keys=True, indent=4))

def refresh_token():
    print("Refreshing token...")
    return spotipy.Spotify(auth_manager=SpotifyOAuth(
        scope=scope, client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri))


def clean_audio_analysis(audio_analysis, id=True):
    # get the acousticness, danceability, energy, instrumentalness, key, liveness, loudness, mode, speechiness, tempo, time_signature, and valence
    my_keys = ['acousticness', 'danceability', 'energy', 'instrumentalness',
        'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo',  'valence']

    if id:
        my_keys.append('id')

    if audio_analysis is None:
        return dict()
    return {key: audio_analysis[key] for key in my_keys}


def get_all_tracks(get_audio_features=False):
    # get all tracks from user library
    offset = 0
    amount = 50
    songs = []
    number = 0
    while True:
        results = spotify.current_user_saved_tracks(
            limit=amount, offset=offset)

        # if we have surpassed the amount of songs saved, break
        if len(results['items']) == 0:
            break

        for _, item in enumerate(results['items']):
            track = item['track']

            if get_audio_features:
                analysis = spotify.audio_features(track['id'])
                songs.append(clean_audio_analysis(analysis[0]))

            else:
                songs.append(track)

            # analysis = spotify.audio_features(track['id'])
            # info = analysis[0]
            # print(json.dumps(analysis, sort_keys=True, indent=4))

            print(number, track['artists'][0]['name'], " – ", track['name'])
            number += 1

        offset += amount

    return songs

# get all of the user's saved tracks
# tracks = get_all_tracks(get_audio_features=True)


def get_songs_from_album(album):
    # get all of the songs from an album
    songs = []
    results = spotify.album_tracks(album['id'])
    songs.extend(results['items'])
    while results['next']:
        results = spotify.next(results)

        # append the id of the songs
        for item in results['items']:
            songs.append(item['id'])

            # print(item['name'])
    return songs


def get_songs_from_artist(artist):
    # get all of the songs from an artist
    songs = []
    results = spotify.artist_albums(artist)
    albums = results['items']
    while results['next']:
        results = spotify.next(results)
        albums.extend(results['items'])

    for album in albums:
        songs.extend(get_songs_from_album(album))

    return songs


# returns a list of all of the songs from all the artists in the user's liked songs
def get_all_songs():
    result = []
    artists = set()
    liked_songs = get_all_tracks()
    for liked_song in liked_songs:
        try:
            # get the id of the artist of the liked song
            artist = liked_song['artists'][0]['id']
            if artist not in artists and artist is not None and artist != '':
                result.extend(get_songs_from_artist(artist))
                artists.add(artist)

                # print the artists name
                print(liked_song['artists'][0]['name'])
                print()

            print("Songs collected: " + str(len(result)))
            print()

        except spotipy.exceptions.SpotifyException:
            global spotify
            spotify = refresh_token()

        if len(result) > 500_000:
            break

    return result


def songs_to_id(song):
    return song['id'] if type(song) == dict else song


def get_songs():
  # print("Getting 100k songs...")
  songs = []
  try:

      songs = get_all_songs()

  except KeyboardInterrupt:
      pass
  # converts songs to song ids
  song_ids = list(map(songs_to_id, songs))
  song_ids = list(set(song_ids))
  
  print("Got the songs and saving them to a file called data_id.json ...")


# save the ids
  with open('data_id.json', 'w', encoding='utf-8') as f:
      json.dump(song_ids, f, ensure_ascii=False, indent=4)



# get a dictionary of the audio features, returns an empty dict if there are none
def get_audio_features_from_id(song_id, id=True):
    analysis = spotify.audio_features(song_id)
    if len(analysis) == 0 or analysis is None:
        return dict()

    return clean_audio_analysis(analysis[0], id=id)


features = []
prev = 0

print("Getting audio features...")
songs_ids = json.load(open('data_id.json', 'r'))

# now get the audio features for every ID

def get_audio_features(song_ids):
  try:
      for idx, id in enumerate(songs_ids):
        try:
          temp = get_audio_features_from_id(id, id=True)
          if temp != dict():
              features.append(temp)
              
          if idx % 100 == 0:
            print(len(features), " songs collected")
        except spotipy.exceptions.SpotifyException:
          global spotify
          spotify = refresh_token()
      
  except KeyboardInterrupt:
    print(prev)


df = pd.read_csv('data_features.csv') 
songs = torch.from_numpy(df.values).float()

print("Read in " + str(len(songs)) + " songs")


def batch(iterable, n=10, shuffle=False, size=None):
    if shuffle:
        iterable = np.random.permutation(iterable)
    l = len(iterable) if size is None else size
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def get_recommended_songs(seed_songs):
    # get recommended songs from the given songs as a seed
    results = spotify.recommendations(seed_tracks=seed_songs, limit=k)
    return results['tracks']


k = 20  # how many songs we produce/recommend
y_s = []
x_s = []
prev = 0
x_ids = []
y_ids = []

print("Preparing data...")
# get the current time
time = time.time()

# get the id's first

for songs in batch(df.index, n=5):
    try:
        if len(songs) != 5:
            prev = prev + len(songs)
            continue

        # gather the y_data (recommendation values, recommendation ids)
        recommended = []
        id_of_recommended = []

        for rec in get_recommended_songs(list(songs)):
            features = get_audio_features_from_id(rec['id'], id=False)

            # append the features of the recommended songs
            recommended.append(list(features.values()))

            id_of_recommended.append(rec['id'])

        y_ids.append(tuple(id_of_recommended))
        y_s.append(tuple(recommended))

        # gather the x_data (seed songs)
        features_x = []
        for song in songs:
            features = get_audio_features_from_id(song, id=False)
            features_x.append(list(features.values()))

        x_ids.append(tuple(songs))
        x_s.append(tuple(features_x))

        prev += 1

        print("Songs processed: " + str(prev))

    except spotipy.exceptions.SpotifyException:
        spotify = refresh_token()
        continue

    except KeyboardInterrupt:
        break
      
    except requests.exceptions.ReadTimeout:
        print(prev)


print("Done preparing data")
x_ids = np.array(x_ids)
y_ids = np.array(y_ids)

x_s = np.array(x_s)
y_s = np.array(y_s)


with open("x_s.npy", 'wb') as f:
    np.save(f, x_s)

with open("y_s.npy", 'wb') as f:
    np.save(f, y_s)


with open("x_ids.npy", 'wb') as f:
    np.save(f, x_ids)

with open("y_ids.npy", 'wb') as f:
    np.save(f, y_ids)


print("Saved data")

print("shape of x_s: ", x_s.shape)
print("shape of y_s: ", y_s.shape)


# print time finished and how long it took
print("Time finished: " + str(time.time()))
print("Time elapsed: " + str(time.time() - time))


def cosine_similarity(a, b):
  return torch.nn.functional.cosine_similarity(a, b, dim=1)


k = 20

import torch.nn

class Recommender(torch.nn.Module):
  def __init__(self, input_shape=(5, 11), feature_size=(20, 11), device='cuda:0') -> None:
      super().__init__()

      # change to basic arch of input_size -> 128 -> 64 -> output_size
      self.songs = songs.to(device)
      self.device = device

      self.encoder = nn.Sequential(
          nn.Flatten(),
          nn.Linear(input_shape[0] * input_shape[1], 128),
          nn.ReLU(True),
          nn.Linear(128, 64),
          nn.ReLU(True),
          nn.Linear(64, feature_size(0) * feature_size(1)))

  def forward(self, x):
    produced = self.encoder(x).to(self.device)
    produced = produced.view(produced.size(0), 20, 11)

    def get_name(id):
      # returns the name of the song given its id
      return spotify.track(id)['name']

    result = torch.zeros(len(produced), k, 11,
                         requires_grad=False).to(self.device)

    for idx, batch in enumerate(produced):
        for idx2, song in enumerate(batch):
            # get the most similar song
          s = torch.argmax(cosine_similarity(song, self.songs))
          result[idx][idx2] = self.songs[s]

    return result

  def recommend(self, seed_songs):
    with torch.no_grad():
      audio_features, names = self.forward(seed_songs)

    return names

