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

# Univariate plots
# box and whisker plots
    # box = lower and upper quartiles
    # whisker = lowest and highest values
    # midline = mean
    # dots = outliers: 3/2 times outside the lower/upper quartile
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
# histograms
dataset.hist()
plt.show()

# Multivariate plots
# scatter plot matrix
scatter_matrix(dataset)
plt.show()

# Split-out dataset into training and validation sets
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
