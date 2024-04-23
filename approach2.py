import os
import urllib.request
import json
import numpy as np
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

# Function to calculate cosine similarity between two vectors
def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    magnitude_A = np.linalg.norm(A)
    magnitude_B = np.linalg.norm(B)
    cosine_similarity = dot_product / (magnitude_A * magnitude_B)
    return cosine_similarity

# Reference review embeddings and ratings
reference_reviews = [
    {"rating": -5, "review": "Absolutely terrible experience. The flight was delayed for hours, the staff was rude, and the food was inedible."},
    {"rating": -3, "review": "Disappointing service overall. While the flight was on time, the seating was uncomfortable, and the amenities were lacking."},
    {"rating": -1, "review": "Slightly below average experience. The flight was uneventful, but the customer service could have been better."},
    {"rating": 0, "review": "Average experience. Nothing outstanding, but nothing terrible either. The flight was on time and the staff was courteous."},
    {"rating": 1, "review": "Satisfactory experience overall. The flight departed and arrived on time, and the staff was helpful."},
    {"rating": 2, "review": "Good service. The flight was comfortable, and the staff was friendly."},
    {"rating": 3, "review": "Very good experience. The flight was smooth, and the amenities provided were satisfactory."},
    {"rating": 4, "review": "Excellent service. The flight was comfortable, and the staff went above and beyond to ensure a pleasant journey."},
    {"rating": 5, "review": "Outstanding experience! The flight was perfect in every way, from the comfortable seating to the delicious food."},
    {"rating": -2, "review": "Below average experience. The flight was delayed, and the service was subpar. Not the worst, but certainly not enjoyable."},
    {"rating": -4, "review": "Poor service all around. The flight was delayed, the seats were uncomfortable, and the staff seemed indifferent. Would not recommend."}
]

# Function to calculate distance between two embeddings
def calculate_distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)

# Function to determine sentiment score of a response
def get_sentiment_score(response_embedding):
    min_distance = float('inf')
    sentiment_score = None
    
    for review in reference_reviews:
        review_embedding = get_openai_embedding(review["review"])
        distance = calculate_distance(response_embedding, review_embedding)
        if distance < min_distance:
            min_distance = distance
            sentiment_score = review["rating"]
    
    return sentiment_score

# Responses for sentiment analysis
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

# Determine sentiment score for each response
for i, response_embedding in enumerate(response_embeddings):
    sentiment_score = get_sentiment_score(response_embedding)
    print(f"Response {i + 1}: Sentiment Score: {sentiment_score}")

#This code retrieves embeddings for the reference reviews and responses using the OpenAI API. It calculates the distance between the embeddings of each response and the embeddings of the reference reviews, assigning the sentiment score of the closest reference review to the response. Finally, it prints out the sentiment score for each response.
