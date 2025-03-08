import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER lexicon
nltk.download('vader_lexicon')
# Initialize VADER SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Function to get sentiment
def get_sentiment(text):
    score = sia.polarity_scores(text)
    if score['compound'] >= 0.05:
        return 'positive'
    elif score['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Apply the sentiment analysis to the 'text' column
data_cleaned_no_duplicates['review_sentiment'] = data_cleaned_no_duplicates['text'].apply(get_sentiment)

# View the results
print(data_cleaned_no_duplicates[['text', 'review_sentiment']].head())
