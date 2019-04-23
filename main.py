# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# -- Data exploration
# Load dataset Iris dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# check dimensions of dataset
print("Dataset shape is {} ".format(dataset.shape))

# take a look at the data 
print(dataset.head(20))

# statistical summaries using describe
print(dataset.describe())

# take a look at the nominal categorical data
print(dataset.groupby('class').size())

# -- Univariate plots
# box and whisker plots
#   box = lower and upper quartiles
#   whisker = lowest and highest values
#   midline = mean
#   dots = outliers: 3/2 times outside the lower/upper quartile
dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, 
                                                        sharey=False)
plt.show()
# histograms
dataset.hist()
plt.show()

# -- Multivariate plots
# scatter plot matrix
scatter_matrix(dataset)
plt.show()

# -- Validation dataset
# Split-out dataset into training and validation sets
# write panda df to array
array = dataset.values
# Split variables into x
X = array[:, 0:4]
# Split categories into Y
Y = array[:, 4]
# Split between training (80%) and validation (20%)
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# -- Test Harness
# using 10-fold cross validation to estimate accuracy, splits datasets into
# 10 parts - train on 9 and test on 1
# seed the data with random numbers to introduce new values
# accuracy is number of correct answers over number of inputs
# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Let’s evaluate 6 different algorithms:
# Logistic Regression (LR)
# Linear Discriminant Analysis (LDA)
# K-Nearest Neighbors (KNN).
# Classification and Regression Trees (CART).
# Gaussian Naive Bayes (NB).
# Support Vector Machines (SVM).

# Create a list of algorithms to be used in models
models = []
models.append(('LR', LogisticRegression(solver='liblinear',
                                        multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, 
                                                cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
