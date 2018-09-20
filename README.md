# d11
from sklearn.datasets import load_iris

iris = load_iris()

print(type(iris))=> <class 'sklearn.utils.Bunch'>
print(iris.data[:2]) =>[[5.1 3.5 1.4 0.2]
                        [4.9 3.  1.4 0.2]]
print(iris.feature_names) => ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
print(iris.target) => [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                       0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                       1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
                       2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
                       2 2]
print(iris.targer_names) => ['setosa' 'versicolor' 'virginica']
print(type(iris.data)) => <class 'numpy.ndarray'>
print(type(iris.target)) => <class 'numpy.ndarray'>
print(iris.data.shape) => (150,4)
print(iris.target.shape) => (150,)
