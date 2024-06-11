import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
 
# Load the pre-trained model and tokenizer
model_name = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
 
def get_embeddings(sentences):
    # Tokenize and encode the sentences
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    # Mean pooling to get sentence embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings
 
def remove_similar_lines(text, similarity_threshold=0.9):
    lines = text.split('\n')
    embeddings = get_embeddings(lines)
    similarities = cosine_similarity(embeddings.detach().numpy())
    
    # Identify and remove similar lines
    to_remove = set()
    for i in range(len(lines)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(lines)):
            if similarities[i, j] > similarity_threshold:
                to_remove.add(j)
 
    # Retain only unique lines
    unique_lines = [line for i, line in enumerate(lines) if i not in to_remove]
    return '\n'.join(unique_lines)
 
# Example usage
with open('qa_chatbot/converted_content/first_chapter.pdf.txt', 'r') as f:
    text = f.read()

print('Original Text')
print('------------------------------')
print(text)
 
shortened_text = remove_similar_lines(text)
print('Shortened Text')
print('------------------------------')
print(shortened_text)