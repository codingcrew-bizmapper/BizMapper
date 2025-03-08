from sklearn.metrics.pairwise import cosine_similarity

# Function to perform search based on user query
def smart_search(query, tfidf_matrix, data, top_n=5):
    query_vec = vectorizer.transform([query])  # Convert query to TF-IDF vector
    similarities = cosine_similarity(query_vec, tfidf_matrix)  # Compute similarity between query and reviews
    
    # Get top N matching reviews
    top_indices = similarities[0].argsort()[-top_n:][::-1]  # Sort to get highest similarity
    
    # Return the top N business names or any other details
    return data.iloc[top_indices]

# Example search query
query = "best cafés for studying"
top_results = smart_search(query, tfidf_matrix, data_cleaned_no_duplicates)
print(top_results[['name', 'text']])
