from dotenv import load_dotenv
import os
import base64
from requests import post, get
import json

load_dotenv()

client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')

def get_token():
    auth_string = client_id + ':' + client_secret
    auth_bytes = auth_string.encode('utf-8')
    auth_base64 = str(base64.b64encode(auth_bytes), 'utf-8')

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": "Basic " + auth_base64,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = { "grant_type": "client_credentials"}
    result = post(url, headers=headers, data=data)
    json_result = json.loads(result.content)
    token = json_result['access_token']
    return token


def get_auth_header(token):
    return { "Authorization": "Bearer " + token }


def search_query(query, token):
    url = "https://api.spotify.com/v1/search?q=" + query + "&type=track&limit=1"
    headers = get_auth_header(token)
    result = get(url, headers=headers)
    json_result = json.loads(result.content)
    uri = json_result['tracks']['items'][0]['uri']
    return uri



def get_features(token, uri):
    uri = uri.strip("spotify:track:")
    url = "https://api.spotify.com/v1/audio-features/" + uri
    headers = get_auth_header(token)
    result = get(url, headers=headers)
    json_result = json.loads(result.content)
    return json_result

def create_dictionary(json_result):
    dictionary = {}
    dictionary['danceability'] = json_result['danceability']
    dictionary['energy'] = json_result['energy']
    dictionary['key'] = json_result['key']
    dictionary['loudness'] = json_result['loudness']
    dictionary['mode'] = json_result['mode']
    dictionary['speechiness'] = json_result['speechiness']
    dictionary['acousticness'] = json_result['acousticness']
    dictionary['instrumentalness'] = json_result['instrumentalness']
    dictionary['liveness'] = json_result['liveness']
    dictionary['valence'] = json_result['valence']
    dictionary['tempo'] = json_result['tempo']
    # dictionary['duration_ms'] = json_result['duration_ms']
    # dictionary['time_signature'] = json_result['time_signature']
    return dictionary




token = get_token()
print(token)



query = "Taylor Swift - Anti-Hero"
result = search_query(query, token)


json_result = get_features(token, result)
print(create_dictionary(json_result))


# print(result)




 