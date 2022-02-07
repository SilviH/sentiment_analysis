import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

review = pd.read_csv('IMDB Dataset.csv')

print('Number of positive reviews in dataset:', len(review[review['sentiment'] == 'positive']))
print('Number of negative reviews in dataset:', len(review[review['sentiment'] == 'negative']))
print('Dataset is balanced')

# Splitting data into train and test set
train, test = train_test_split(review, test_size=0.33, random_state=17)

# Set the independent variables and dependent variables
train_x, train_y = train['review'], train['sentiment']
test_x, test_y = test['review'], test['sentiment']

# TF-IDF - Transform text to vectors
tfidf = TfidfVectorizer(stop_words='english')
train_x_vector = tfidf.fit_transform(train_x)
print(
    'The train_x_vector is a sparse matrix with a shape of',
    train_x_vector.shape[0],
    'reviews and', train_x_vector.shape[1],
    'words (whole vocabulary used in the reviews).'
)
test_x_vector = tfidf.transform(test_x)

# Classification algorithms - Support Vector Machines (SVM)
svc = SVC(kernel='linear')
svc.fit(train_x_vector, train_y)

test_reviews = [
    'A good movie',
    'An excellent movie',
    'I did not like this movie at all',
    'Bad intention but good execution'
]
for n in test_reviews:
    print(svc.predict(tfidf.transform([n])))

# F1 Score (SVM only)
print(
    'The scores obtained for positive and negative labels are:',
    f1_score(test_y, svc.predict(test_x_vector), labels=['positive', 'negative'], average=None)
)
# Confusion Matrix
print('Confusion matrix:\n', confusion_matrix(test_y,  svc.predict(test_x_vector),  labels=['positive', 'negative']))