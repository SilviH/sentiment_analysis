import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

review = pd.read_csv('IMDB Dataset.csv')

# Creation of smaller and balanced dataset to train the model faster in first steps
review_positive = review[review['sentiment'] == 'positive'][:1000]
review_negative = review[review['sentiment'] == 'negative'][:1000]
review_bal = pd.concat([review_positive, review_negative])

# Splitting data into train and test set
train, test = train_test_split(review_bal, test_size=0.33, random_state=17)

# Data overview
print('Number of positive reviews in train:', len(train[train['sentiment'] == 'positive']))
print('Number of negative reviews in train:', len(train[train['sentiment'] == 'negative']))
print('Number of positive reviews in test:', len(test[test['sentiment'] == 'positive']))
print('Number of negative reviews in test:', len(test[test['sentiment'] == 'negative']))

# Set the independent variables (review) and dependent variables (sentiment) within the train and test set
train_x, train_y = train['review'], train['sentiment']
test_x, test_y = test['review'], test['sentiment']

# Text representation - TF-IDF method
tfidf = TfidfVectorizer(stop_words='english')

# TF-IDF
# Fitting (calculates parameters to standardize data) and transformation (applies the parameters to any particular
# set of examples) of train
train_x_vector = tfidf.fit_transform(train_x)
print(
    'The train_x_vector is a sparse matrix with a shape of',
    train_x_vector.shape[0],
    'reviews and', train_x_vector.shape[1],
    'words (whole vocabulary used in the reviews).'
)

# TF-IDF
# Display of the matrix
print(pd.DataFrame.sparse.from_spmatrix(train_x_vector, index=train_x.index, columns=tfidf.get_feature_names_out()))

# TF-IDF
# Transformation of test (we don’t need to fit tfidf again)
test_x_vector = tfidf.transform(test_x)
