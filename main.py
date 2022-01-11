import pandas as pd

review = pd.read_csv('IMDB Dataset.csv')

# Creation of smaller and balanced dataset to train the model faster in first steps
review_positive = review[review['sentiment'] == 'positive'][:1000]
review_negative = review[review['sentiment'] == 'negative'][:1000]
review_imb = pd.concat([review_positive, review_negative])
