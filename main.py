import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

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

# Transformation of test (we don’t need to fit tfidf again)
test_x_vector = tfidf.transform(test_x)

# Classification algorithms
test_reviews = [
    'A good movie',
    'An excellent movie',
    'I did not like this movie at all',
    'Bad intention but good execution'
]
gnb = None


def test_method(classification_method):
    for n in test_reviews:
        if classification_method == gnb:
            print(classification_method.predict((tfidf.transform([n])).toarray()))
        else:
            print(classification_method.predict(tfidf.transform([n])))
    print()


# Classification algorithms - Support Vector Machines (SVM)
svc = SVC(kernel='linear')
svc.fit(train_x_vector, train_y)
test_method(svc)

# Classification algorithms - Decision Tree
dec_tree = DecisionTreeClassifier()
dec_tree.fit(train_x_vector, train_y)
test_method(dec_tree)

# Classification algorithms - Naive Bayes
gnb = GaussianNB()
gnb.fit(train_x_vector.toarray(), train_y)
test_method(gnb)

# Classification algorithms - Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(train_x_vector, train_y)
test_method(log_reg)

# Model Evaluation
# Mean Accuracy - used when the True Positives and True negatives are more important.
print('SVM mean accuracy:', svc.score(test_x_vector, test_y))
print('Decision Tree mean accuracy:', dec_tree.score(test_x_vector, test_y))
print('Naive Bayes mean accuracy:', gnb.score(test_x_vector.toarray(), test_y))
print('Logistic Regression mean accuracy:', log_reg.score(test_x_vector, test_y))

# F1 Score (SVM only)
print(
    'The scores obtained for positive and negative labels are:',
    f1_score(test_y, svc.predict(test_x_vector), labels=['positive', 'negative'], average=None)
)

# Classification report (calculates both accuracy and F1-score and also others)
print(classification_report(test_y, svc.predict(test_x_vector), labels=['positive', 'negative']))

# Confusion Matrix - table that reports the number of false positives, negatives and true positives, negatives
print(confusion_matrix(test_y,  svc.predict(test_x_vector),  labels=['positive', 'negative']))

