from transformers import pipeline

# Load the text generation pipeline with BERT
generator = pipeline("text-generation", model="bert-base-uncased", tokenizer="bert-base-uncased")

# Input sentence
input_sentence = "The food was awful it looked awful and tasted awful I will be sending a letter and will attach pictures."

# Tokenize the input sentence
tokens = input_sentence.split()

# List to store generated sentences
generated_sentences = []

# Generate text for each part of the input sentence
for i in range(len(tokens)):
    # Join tokens up to the current index to form a partial sentence
    partial_sentence = ' '.join(tokens[:i+1])
    # Generate text for the partial sentence
    generated_text = generator(partial_sentence, max_length=50, num_return_sequences=1, do_sample=True)[0]['generated_text']
    # Append the generated text to the list
    generated_sentences.append(generated_text.strip())

# Print the generated sentences
for sentence in generated_sentences:
    print(sentence)
