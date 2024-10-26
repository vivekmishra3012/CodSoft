import pandas as pd
import sys
print(sys.executable)
print(sys.version)
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample datasets
movies = pd.DataFrame({
    'title': ['The Shawshank Redemption', 'The Godfather', 'The Dark Knight', 'Pulp Fiction', 'Forrest Gump'],
    'description': [
        'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
        'The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.',
        'When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.',
        'The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.',
        'The presidencies of Kennedy and Johnson, the Vietnam War, the Watergate scandal and other historical events unfold from the perspective of an Alabama man with an IQ of 75, whose only desire is to be reunited with his childhood sweetheart.'
    ]
})

books = pd.DataFrame({
    'title': ['To Kill a Mockingbird', '1984', 'Pride and Prejudice', 'The Great Gatsby', 'The Catcher in the Rye'],
    'description': [
        'The story of racial injustice and the loss of innocence in the American South during the Great Depression.',
        'A dystopian social science fiction novel and cautionary tale set in a totalitarian society.',
        'A romantic novel of manners that follows the character development of Elizabeth Bennet.',
        'A novel about the American Dream, decadence, idealism, and resistance to change.',
        'A story of teenage angst and alienation, exploring complex issues of innocence, identity, and connection.'
    ]
})

products = pd.DataFrame({
    'title': ['iPhone 12', 'Samsung Galaxy S21', 'Sony PlayStation 5', 'Nintendo Switch', 'Amazon Echo Dot'],
    'description': [
        'A powerful smartphone with 5G capabilities, advanced camera system, and A14 Bionic chip.',
        'A flagship Android smartphone featuring a high-refresh-rate display and versatile camera setup.',
        'Next-generation gaming console with ray tracing, fast loading, and immersive DualSense controller.',
        'Hybrid gaming console that can be played on-the-go or docked for TV play.',
        'Smart speaker with Alexa, offering voice control for music, smart home devices, and more.'
    ]
})

# Combine all datasets
all_items = pd.concat([movies, books, products], ignore_index=True)

# Generate embeddings for all items
all_embeddings = model.encode(all_items['description'].tolist())

def generate_simple_description(item):
    # Split the item name into words
    words = item.lower().split()
    # Create a simple description using the words
    description = f"An item related to {', '.join(words)}."
    return description

def get_recommendations(query, top_n=5):
    # Generate description for the query if it's not in our dataset
    if query not in all_items['title'].values:
        query_description = generate_simple_description(query)
    else:
        query_description = all_items[all_items['title'] == query]['description'].values[0]

    # Generate embedding for the query
    query_embedding = model.encode([query_description])

    # Calculate cosine similarity between query and all items
    similarities = cosine_similarity(query_embedding, all_embeddings)

    # Get indices of top N similar items
    top_indices = similarities[0].argsort()[-top_n:][::-1]

    # Return top N recommendations
    recommendations = all_items.iloc[top_indices]
    return recommendations[['title', 'description']]

def main():
    while True:
        query = input("Enter a movie, book, or product (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break

        print(f"\nRecommendations for '{query}':")
        recommendations = get_recommendations(query)
        for i, (_, row) in enumerate(recommendations.iterrows(), 1):
            print(f"{i}. {row['title']}")
            print(f"   {row['description']}\n")

if __name__ == "__main__":
    main()