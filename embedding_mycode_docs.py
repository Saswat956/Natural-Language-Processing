import os
import urllib.request
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

def get_openai_embedding(prompt):
    """
    Retrieves embeddings for a given text prompt using the OpenAI API.

    Parameters:
    - prompt (str): The text prompt for which embeddings are to be retrieved.

    Returns:
    - np.array: An array containing the embeddings for the given text prompt.
    """
    # Function body...

def visualize_pca_2d(embeddings, words):
    """
    Visualizes word embeddings in 2D space using PCA.

    Parameters:
    - embeddings (np.array): An array containing the word embeddings.
    - words (list): A list of words corresponding to the embeddings.

    Returns:
    - None
    """
    # Function body...

def scale_score(score):
    """
    Scales the sentiment score to the range -5 to 5.

    Parameters:
    - score (float): The sentiment score to be scaled.

    Returns:
    - float: The scaled sentiment score.
    """
    # Function body...

def cosine_similarity(A, B):
    """
    Calculates the cosine similarity between two vectors.

    Parameters:
    - A (np.array): The first vector.
    - B (np.array): The second vector.

    Returns:
    - float: The cosine similarity between the two vectors.
    """
    # Function body...

def calculate_distance(point1, point2):
    """
    Calculates the Euclidean distance between two points in n-dimensional space.

    Parameters:
    - point1 (tuple): The coordinates of the first point.
    - point2 (tuple): The coordinates of the second point.

    Returns:
    - float: The Euclidean distance between the two points.
    """
    # Function body...

def get_sentiment_score(review):
    """
    Determines the sentiment score for a given review.

    Parameters:
    - review (str): The text of the review.

    Returns:
    - float: The sentiment score in the range of -5 to 5.
    """
    # Function body...

# Loading OpenAI key
with open("openai_key", "r") as file:
    openai_key = file.read()

# Setting up the environment variable
os.environ["OPENAI_KEY"] = openai_key

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

# Determine sentiment score of each response
for i, response in enumerate(responses):
    print(f"Response {i + 1}: {response}")
    sentiment_score = get_sentiment_score(response)
    print(f"Sentiment Score: {sentiment_score:.2f}")
    print()


This code retrieves word embeddings for text prompts using the OpenAI API.
It visualizes the word embeddings in 2D space using Principal Component Analysis (PCA).
Sentiment scores are calculated for responses provided in the responses list.
The sentiment scores are scaled to the range of -5 to 5, where negative scores indicate negative sentiment and positive scores indicate positive sentiment.
Finally, the sentiment scores for each response are printed out.
