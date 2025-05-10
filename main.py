import string
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import requests
import matplotlib.pyplot as plt

def get_text(url):
    """
    Download text from the given URL
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error downloading text: {e}")
        return None

def remove_punctuation(text):
    """
    Remove punctuation from text
    """
    return text.translate(str.maketrans("", "", string.punctuation))

def map_function(word):
    """
    Map function for MapReduce
    """
    return word.lower(), 1

def shuffle_function(mapped_values):
    """
    Shuffle function for MapReduce
    """
    shuffled = defaultdict(list)
    for key, value in mapped_values:
        shuffled[key].append(value)
    return shuffled.items()

def reduce_function(key_values):
    """
    Reduce function for MapReduce
    """
    key, values = key_values
    return key, sum(values)

def map_reduce(text):
    """
    Execute MapReduce process on text
    """
    # Remove punctuation and split into words
    text = remove_punctuation(text)
    words = text.split()

    # Parallel Mapping
    with ThreadPoolExecutor() as executor:
        mapped_values = list(executor.map(map_function, words))

    # Shuffle
    shuffled_values = shuffle_function(mapped_values)

    # Parallel Reduction
    with ThreadPoolExecutor() as executor:
        reduced_values = list(executor.map(reduce_function, shuffled_values))

    return dict(reduced_values)

def visualize_top_words(word_freq, top_n=10):
    """
    Visualize top N words by frequency
    """
    # Sort words by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    top_words = sorted_words[:top_n]

    # Prepare data for plotting
    words = [word for word, _ in top_words]
    frequencies = [freq for _, freq in top_words]

    # Create bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(words, frequencies)
    plt.title(f'Top {top_n} Most Frequent Words')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def main():
    # URL of the text to analyze
    url = "https://gutenberg.net.au/ebooks01/0100021.txt"
    
    # Get text from URL
    text = get_text(url)
    if not text:
        return

    # Apply MapReduce to get word frequencies
    word_frequencies = map_reduce(text)

    # Visualize top 10 words
    visualize_top_words(word_frequencies, 10)

if __name__ == '__main__':
    main() 