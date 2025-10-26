import nltk
from nltk import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords

# Download all required data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

text_name = 'the_pearl.txt'
scrubbed_words = []
filter_proper_nouns = True
filter_stop_words = False # e.g., "a", "the", "and"

# Your list of proper nouns to filter out
# scrubbed_words = [
#     'Kino', 'Juana', 'Coyotito', 'Juan', 'Tomas',
#     'La', 'Paz', 'Gulf', 'California', 'Mexico',
# ]
scrubbed_words_lower = [name.lower() for name in scrubbed_words]

# Read the text file
with open('the_pearl.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Tokenize
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

# Get top N words
top_n = 1024  # Change this to whatever you want
top_words = fdist.most_common(top_n)

# Display results
print(f"Total words analyzed: {len(filtered_tokens)}")
print(f"\nTop {min(top_n, len(top_words))} words by frequency:\n")
for word, freq in top_words[:50]:  # Show first 50
    print(f"{word:20} {freq}")

# Save to file
with open('top_words.txt', 'w', encoding='utf-8') as f:
    for word, freq in top_words:
        f.write(f"{word}\t{freq}\n")