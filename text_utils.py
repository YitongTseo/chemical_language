import nltk
from nltk import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import numpy as np

# Download all required data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

def get_top_words(
    text,
    top_n,
    scrubbed_words,
    filter_stop_words,
):
    scrubbed_words_lower = [name.lower() for name in scrubbed_words]
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    if filter_stop_words:
        scrubbed_words_lower = scrubbed_words_lower + list(stop_words)

    filtered_tokens = [
        word for word in tokens 
        if word.isalpha()
        and word not in scrubbed_words_lower  
    ]

    # Calculate frequency distribution
    fdist = FreqDist(filtered_tokens)
    top_words = fdist.most_common(top_n)
    
    print(f"Total words analyzed: {len(filtered_tokens)}")
    print(f"\nTop {min(top_n, len(top_words))} words by frequency:\n")
    for word, freq in top_words[:50]: 
        print(f"{word:20} {freq}")

    return top_words

def create_numpy_transition_matrix(text, top_words_list):
    """
    Creates a NumPy transition probability matrix P (N x N) for the top_n words.
    """
    # 1. Setup Index Mapping
    top_words_voc = [word for word, count in top_words_list]
    N = len(top_words_voc)
    word_to_index = {word: i for i, word in enumerate(top_words_voc)}
    
    # 2. Tokenize and Filter
    tokens = [word.lower() for word in nltk.word_tokenize(text)]
    
    # 3. Initialize Count Matrix
    count_matrix = np.zeros((N, N), dtype=int)
    # 4. Populate Counts
    for i in range(len(tokens) - 1):
        current_word = tokens[i]
        next_word = tokens[i+1]
        
        # Check if both words are in the top_words set
        if current_word in word_to_index and next_word in word_to_index:
            row_idx = word_to_index[current_word]
            col_idx = word_to_index[next_word]
            count_matrix[row_idx, col_idx] += 1

    # 5. Normalize (Create Probability Matrix P)
    # Sum across rows (axis=1) to get total outgoing transitions for each word
    row_sums = count_matrix.sum(axis=1, keepdims=True)
    
    # Handle rows where the top word doesn't have any subsequent top words
    # to prevent division by zero; setting those rows to 1 avoids NaNs.
    row_sums[row_sums == 0] = 1 
    
    # Divide the counts by the row sums to get probabilities
    probability_matrix = count_matrix / row_sums
    
    return probability_matrix, top_words_voc

# P, vocabulary = create_numpy_transition_matrix(text, top_words)
# P[i, j] is the probability of word j following word i, 
# where i and j are the indices in the vocabulary list.