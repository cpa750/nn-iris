import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

print(dataset.shape)
print(dataset.head(10))

dataset.hist()
#plt.show()

scatter_matrix(dataset)
#plt.show()

array = dataset.values
X, Y = array[:, :4], array[:, 4]
print(array[0, 4])
print(array[50, 4])
print(array[149, 4])

