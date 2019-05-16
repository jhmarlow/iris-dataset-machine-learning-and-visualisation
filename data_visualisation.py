
from google.cloud import bigquery

# import modules
import pandas as pd
# import matplotlib.pyplot as plt
import seaborn as sns

# set sns style
sns.set(style='white', color_codes=True)

# -- Data exploration
# Load dataset Iris dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
iris_dataset = pd.read_csv(url, names=names)

# take a look
print(iris_dataset.head())

print(iris_dataset['class'].value_counts())

iris_dataset.plot(kind="scatter", x="sepal-length", y="sepal-width")