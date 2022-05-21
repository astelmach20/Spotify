from sklearn.model_selection import train_test_split
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
from torch.nn.modules.activation import ReLU
from torch.nn.functional import normalize
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch import nn
import torch
from numpy.linalg import norm
from numpy import dot
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util
import json
import numpy as np

print("finished importing")

client_id = 'a054865bcfda4b029dad65f46a058b19'
client_secret = os.environ['SPOTIPY_CLIENT_SECRET']

scope = "user-library-read"
username = 'theswifter01'
redirect_uri = 'http://example.com'

token = util.prompt_for_user_token(
    username, scope, client_id, client_secret, redirect_uri)

spotify = spotipy.Spotify(auth=token)
user = spotify.current_user()

df = pd.read_csv('data_features.csv')
songs = torch.from_numpy(df.values).float()


def clean_audio_analysis(audio_analysis, id=True):
    # get the acousticness, danceability, energy, instrumentalness, key, liveness, loudness, mode, speechiness, tempo, time_signature, and valence
    my_keys = ['acousticness', 'danceability', 'energy', 'instrumentalness',
               'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence']

    if id:
        my_keys.append('id')

    if audio_analysis is None:
        return dict()
    return {key: audio_analysis[key] for key in my_keys}


# get a dictionary of the audio features, returns an empty dict if there are none


def get_audio_features_from_id(song_id, id=True):
    analysis = spotify.audio_features(song_id)
    if len(analysis) == 0 or analysis is None:
        return dict()

    return clean_audio_analysis(analysis[0], id=id)


def get_name(id):
    # returns the name of the song given its id
    return spotify.track(id)['name']


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


k = 20

x_s = np.load("x_s.npy")
y_s = np.load("y_s.npy")


def cosine_similarity(a, b, axis=1):
    return torch.nn.functional.cosine_similarity(a, b, dim=axis)


class Recommender(torch.nn.Module):
    def __init__(self, input_shape=(5, 11), feature_size=(20, 11), device='cuda:0') -> None:
        super().__init__()

        # change to basic arch of input_size -> 128 -> 64 -> output_size
        self.songs = songs.to(device)
        self.device = device
        self.feature_size = feature_size

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape[0] * input_shape[1], 128),
            nn.Linear(128, 64),
            nn.LeakyReLU(True),
            nn.Dropout(p=.02, inplace=True),
            nn.Linear(64, feature_size[0] * feature_size[1]))

    # produces a (batch size x k x 11) tensor of songs that we *should* predict/create to the user
    # these values are hypothetical by the net so we actually recommend the most similar songs to what the net produced
    # via the cosine similarity of songs in our db
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)

        produced = self.net(x).to(self.device)
        return produced.view(produced.size(0), self.feature_size[0], self.feature_size[1])

    # returns a (batch x k x 11) tensor of numeric song values to validate with spotify's recommendations
    # these numeric song values are based upon the database of songs that we have

    def validation(self, x):
        with torch.no_grad():
            produced = self.forward(x)

        result = torch.zeros(len(produced), k, 11,
                             requires_grad=False).to(self.device)

        for idx, batch in enumerate(produced):
            for idx2, song in enumerate(batch):
                # get the most similar song
                s = torch.argmax(cosine_similarity(song, self.songs))
                result[idx][idx2] = self.songs[s]

        return result

    # returns a list of song names
    def recommend(self, song_ids):
        temp = []
        if type(song_ids[0]) == str:
            for i in song_ids:
                features = get_audio_features_from_id(i, id=False)
                temp.append(list(features.values()))

            song_ids = temp

        song_ids = torch.tensor(song_ids).float().to(self.device)

        with torch.no_grad():
            produced = self.forward(song_ids)

        names = []
        for song in produced[0]:
            # get the highest similar songs
            s = torch.argmax(cosine_similarity(song, self.songs)).cpu().numpy()

            id = df.index[s]
            names.append(get_name(id))

        return list(set(names))


mse = nn.MSELoss()


class VariationalRecommender(torch.nn.Module):
    def __init__(self, input_shape=(5, 11), feature_size=(20, 11), device='cuda:0') -> None:
        super().__init__()

        # change to basic arch of input_size -> 128 -> 64 -> output_size
        self.songs = songs.to(device)
        self.device = device
        self.feature_size = feature_size

        self.stats = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape[0] * input_shape[1], 64),
            nn.LeakyReLU(True),
            nn.Linear(64, 64),
            nn.LeakyReLU(True),
            nn.Linear(64, 2))

        self.recommender = nn.Sequential(
            nn.Linear(1, 128),
            nn.LeakyReLU(True),
            nn.Linear(128, 64),
            nn.LeakyReLU(True),
            nn.Dropout(.2, inplace=True),
            nn.Linear(64, feature_size[0] * feature_size[1]))

    def forward(self, x):

        if x.dim() == 2:
            x = x.unsqueeze(0)

        # get the mean and std
        z = self.stats(x)
        mean, std = torch.split(z, 1, dim=1)

        # reparameterization trick
        def reparam(mean, std):
            e = Normal(loc=torch.zeros_like(mean),
                       scale=torch.ones_like(std)).sample()
            return mean + std * e

        sampled = reparam(mean, std)

        produced = self.recommender(sampled)

        return (mean, torch.exp(std)), produced.view(produced.size(0), self.feature_size[0], self.feature_size[1])

    def validation(self, x):
        with torch.no_grad():
            (mean, std), produced = self.forward(x)

        result = torch.zeros(len(produced), k, 11,
                             requires_grad=False).to(self.device)

        for idx, batch in enumerate(produced):
            for idx2, song in enumerate(batch):
                # get the most similar song
                s = torch.argmax(cosine_similarity(song, self.songs))
                result[idx][idx2] = self.songs[s]

        return (mean, std), result

    # returns a list of song names
    def recommend(self, song_ids):
        temp = []
        if type(song_ids[0]) == str:
            for i in song_ids:
                features = get_audio_features_from_id(i, id=False)
                temp.append(list(features.values()))

            song_ids = temp

        song_ids = torch.tensor(song_ids).float().to(self.device)

        with torch.no_grad():
            _, produced = self.forward(song_ids)

        names = []
        for song in produced[0]:
            idx = 0
            s = torch.argsort(cosine_similarity(song, self.songs), descending=True)
            while len(names) < 20:
                # get the _idx_ most similar songs
                id = df.index[s[idx].cpu().numpy()]
                name = get_name(id)

                if name not in names:
                    names.append(name)
                    
                    break

                else:
                    idx += 1

        return names


class NegativeELBOLoss:
    def __init__(self, beta=1):
        self.beta = beta

    def __call__(self, outs, x):
        (z_mean, z_std), x_hat = outs
        q = Normal(z_mean, z_std)
        p = Normal(torch.zeros_like(z_mean), torch.ones_like(z_std))
        return mse(x_hat, x) + self.beta * kl_divergence(q, p).mean()


def validation_loop(model, loss_fn, x_val, y_val, batch_size=32,
                    device='cuda:0'):
    x_val = x_val.to(device)
    y_val = y_val.to(device)

    val_loss = 0
    for i in range(0, len(x_val), batch_size):
        x = x_val[i:i + batch_size]
        y = y_val[i:i + batch_size]

        with torch.no_grad():
            outs = model(x)
            val_loss += loss_fn(outs, y).item()

    return val_loss / len(x_val)


def train_model(model, optimizer, loss_fn, x_train, y_train, x_val=None, y_val=None, validate_every=1,
                validation_songs=False, epochs=20, batch_size=32, device='cuda:0'):
    validation = x_val is not None and y_val is not None and len(
        x_val) == len(y_val)

    if validation:
        # shuffle the validation set
        indices = np.random.permutation(len(x_val))
        x_val = x_val[indices]
        y_val = y_val[indices]

        # convert to tensors
        x_val = torch.Tensor(x_val).to(device)
        y_val = torch.Tensor(y_val).to(device)

        x_val = normalize(x_val)
        y_val = normalize(y_val)

    model = model.to(device)
    train_losses = []
    train_epochs = []

    val_song_losses = []
    val_song_epochs = []

    val_losses = []
    val_epochs = []

    # shuffle the x_train and y_train data with the same indices
    indices = np.random.permutation(len(y_train))
    x_train = x_train[indices]
    y_train = y_train[indices]

    # convert x_train and Y_train to tensors
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()

    x_train = x_train.to(device)
    y_train = y_train.to(device)

    x_train = normalize(x_train)
    y_train = normalize(y_train)

    for epoch in range(1, epochs + 1):
        train_loss = 0
        validation_loss = 0

        # loop through entire training set and select a random batch
        for i in range(0, len(y_train), batch_size):
            # get the batch
            x_batch = x_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            # zero the gradients
            optimizer.zero_grad()

            # forward pass
            y_pred = model(x_batch)

            # calculate loss
            loss = loss_fn(y_pred, y_batch)
            train_loss += loss.item()

            # backward pass
            loss.backward()

            # update weights
            optimizer.step()

        if validation_songs and epoch % validate_every == 0:
            # validation loss
            validation_pred = model.validation(x_batch)
            validation_loss += loss_fn(validation_pred, y_batch).item()
            validation_loss /= len(x_train) // batch_size
            val_song_losses.append(validation_loss)
            val_song_epochs.append(epoch)
            print("Epoch: " + str(epoch) +
                  " Validation Song Loss: " + str(validation_loss))

        if validation and epoch % validate_every == 0:
            # validation loss
            val_losses.append(validation_loop(
                model, loss_fn, x_val, y_val, batch_size, device))
            val_epochs.append(epoch)
            # print out the validation
            print("Epoch: " + str(epoch) +
                  " Validation Loss: " + str(val_losses[-1]))

        train_loss /= (len(x_train) // batch_size)
        train_losses.append(train_loss)
        train_epochs.append(epoch)

        print("Epoch: " + str(epoch) + " Training Loss: " + str(train_loss))
        print()

    t_l = np.array(train_losses)
    t_e = np.array(train_epochs)

    v_s_l = np.array(val_song_losses)
    v_s_e = np.array(val_song_epochs)

    v_l = np.array(val_losses)
    v_e = np.array(val_epochs)

    return t_e, t_l, v_l, v_e, v_s_l, v_s_e


# split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(
    x_s, y_s, test_size=0.2, random_state=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Recommender(device=device)
optimizer = Adam(model.parameters(), lr=0.001)
t_e, t_l, v_l, v_e, v_s_l, v_s_e = train_model(
    model, optimizer, mse, x_train, y_train, x_val, y_val, epochs=100, device=device, validation_songs=True,
    validate_every=5)

plt.plot(t_e, t_l, label='Training Loss')
plt.legend()
plt.show()

plt.plot(v_e, v_l, label='Validation Loss', color='orange')
plt.legend()
plt.show()

plt.plot(v_s_e, v_s_l, label='Validation Song Loss', color='red')
plt.legend()
plt.show()

test_songs = ['0nrRP2bk19rLc0orkWPQk2', '5UqCQaDshqbIk3pkhy4Pjg', '2cYqizR4lgvp4Qu6IQ3qGN', '0tdCy39PgWN8LFWu34ORn3',
              '3eekarcy7kvN4yt5ZFzltW']
print(model.recommend(test_songs))


model = VariationalRecommender(device=device)
optimizer = Adam(model.parameters(), lr=0.0001)
t_e, t_l, v_l, v_e, v_s_l, v_s_e = train_model(
    model, optimizer, NegativeELBOLoss(beta=1e4), x_train, y_train, x_val, y_val, epochs=100, device=device,
    validation_songs=True, validate_every=5)

plt.plot(t_e, t_l, label='Training Loss')
plt.legend()
plt.show()

# plot this in orange
plt.plot(v_e, v_l, label='Validation Loss', color='orange')
plt.legend()
plt.show()

plt.plot(v_s_e, v_s_l, label='Validation Song Loss', color='red')
plt.legend()
plt.show()

# search for "wake me up"
test_songs = ['0nrRP2bk19rLc0orkWPQk2', '5UqCQaDshqbIk3pkhy4Pjg', '2cYqizR4lgvp4Qu6IQ3qGN', '0tdCy39PgWN8LFWu34ORn3',
              '3eekarcy7kvN4yt5ZFzltW']
print(model.recommend(test_songs))
