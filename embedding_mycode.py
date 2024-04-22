import os
import urllib.request
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Loading OpenAI key
with open("openai_key", "r") as file:
    openai_key = file.read()

# Setting up the environment variable
os.environ["OPENAI_KEY"] = openai_key

# Function to make API request for embeddings
def get_openai_embedding(prompt):
    url = 'https://api.openai.com/v1/embeddings'
    headers = {
        'Authorization': f'Bearer {openai_key}',
        'Content-Type': 'application/json',
    }
    data = {
        "input": prompt,    
        "model": "text-embedding-ada-002"
    }
    data = json.dumps(data).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers=headers, method='POST')
    
    try:
        response = urllib.request.urlopen(req)
        response_data = json.loads(response.read().decode('utf-8'))
        return np.array(response_data['data'][0]['embedding'])
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e}")
        return None

# Function to visualize word embeddings in 2D space
def visualize_pca_2d(embeddings, words):
    pca_2d = PCA(n_components=2)
    embeddings_2d = pca_2d.fit_transform(embeddings)

    plt.figure(figsize=(10, 6))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], marker='o')
    for i, word in enumerate(words):
        plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]))

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("2D Visualization of Word Embeddings")
    plt.grid(True)
    plt.show()

# Function to determine sentiment score of a review
def get_sentiment_score(review):
    embed_review = get_openai_embedding(review)
    dist_pos = calculate_distance(embed_review, embd_positive)
    dist_neg = calculate_distance(embed_review, embd_neg)
    cosine_sim_pos = cosine_similarity(embed_review, embd_positive)
    cosine_sim_neg = cosine_similarity(embed_review, embd_neg)
    
    score_pos = dist_pos * cosine_sim_pos
    score_neg = dist_neg * cosine_sim_neg
    
    # Scale the scores to the range -5 to 5
    score_pos_scaled = scale_score(score_pos)
    score_neg_scaled = scale_score(score_neg)
    
    if score_pos_scaled > score_neg_scaled:
        return score_pos_scaled
    else:
        return -score_neg_scaled


# Function to determine sentiment of a review
def is_positive(review):
    embed_review = get_openai_embedding(review)
    dist_pos = calculate_distance(embed_review, embd_positive)
    dist_neg = calculate_distance(embed_review, embd_neg)
    if dist_pos < dist_neg:
        print("It is a positive review")
        return True
    else:
        print("It is a negative review")
        return False

# Function to calculate cosine similarity between two vectors
def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    magnitude_A = np.linalg.norm(A)
    magnitude_B = np.linalg.norm(B)
    cosine_similarity = dot_product / (magnitude_A * magnitude_B)
    return cosine_similarity

# Sentences for sentiment analysis
responses = [
    "Flight was on time.",
    "It was crowded and hot and humidity was awful. The gate area didn't have enough seats and the flight was delayed and your customer were very uncomfortable.",
    "The food was awful it looked awful and tasted awful I will be sending a letter and will attach pictures",
    "Crepes and pasta",
    "I live united but this trip I was disappointed",
    "On time departure and arrival, effective boarding. Negative one restroom was out of order."
]

# Get embeddings for responses
response_embeddings = [get_openai_embedding(response) for response in responses]

# Visualize embeddings in 2D space
visualize_pca_2d(np.array(response_embeddings), responses)

# Define positive and negative embeddings
embd_positive = get_openai_embedding('positive')
embd_neg = get_openai_embedding('negative')

# Determine sentiment of each response
for i, response in enumerate(responses):
    print(f"Response {i + 1}: {response}")
    is_positive(response)
    print()
