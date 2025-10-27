import nltk
from nltk import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords

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