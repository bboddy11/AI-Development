import random
from textblob import TextBlob
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')

# Створимо 100 штучних відгуків (для демонстрації)
positive_reviews = [
    "I love this product!", "Absolutely fantastic!", "Great value for money.",
    "Exceeded my expectations!", "Highly recommend this!", "I am very satisfied.",
    "Five stars!", "Amazing service and quality.", "It works perfectly.", "Incredible experience."
]

negative_reviews = [
    "Terrible experience.", "I hate it.", "Very disappointed.",
    "It broke after one use.", "Not worth the money.", "Would not recommend.",
    "Worst product ever.", "Cheap and useless.", "Waste of time.", "Extremely bad quality."
]

neutral_reviews = [
    "It’s okay.", "Nothing special.", "It works as expected.",
    "Average experience.", "Not good, not bad.", "Fair enough.",
    "I have no strong feelings.", "It’s decent.", "So-so.", "Neutral review."
]

# Генеруємо 100 випадкових відгуків
all_reviews = random.choices(positive_reviews, k=34) + \
              random.choices(negative_reviews, k=33) + \
              random.choices(neutral_reviews, k=33)
random.shuffle(all_reviews)

# Аналіз тональності
sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}

for review in all_reviews:
    blob = TextBlob(review)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        sentiment_counts['positive'] += 1
    elif polarity < -0.1:
        sentiment_counts['negative'] += 1
    else:
        sentiment_counts['neutral'] += 1

# Виведення результатів
print("Sentiment distribution:")
for k, v in sentiment_counts.items():
    print(f"{k.capitalize()}: {v}")

# Побудова діаграми
plt.figure(figsize=(7, 5))
plt.bar(sentiment_counts.keys(), sentiment_counts.values())
plt.title("Sentiment Analysis of 100 Reviews")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
