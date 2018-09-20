# d11
#KNeighbors
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

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

X = iris.data
y = iris.target
knn = KNeighbors(n_neighbors=1)
knn.fit(X,y)
print(knn.predict([5,4,3,2],)) => [1]
X_test = ([5,4,3,2],[3,5,4,2])
print(knn.predict(X_test)) => [1,1]    #labels
print(knn.predict_proba(X_test)) => [[0.  1.  0. ]                
                                     [0.  0.8 0.2]]            #X_test
print(knn.score(X,y)) => 0.9666666666666667
print(knn.kneighbors(X[:1,:],n_neighbors=1)) => (array([[0.]]), array([[0]], dtype=int32))

#LogisticRegression
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

iris = load_iris()
X = iris.data
y = iris.target
log = LogisticRegression()
log.fit(X,y)
y_pred = log.predict(X)
print(y_pred)
print(metrics.accuracy_score(y,y_pred)) => 0.96

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=4)
print(X_train.shape) => (90,4)
print(X_test.shape) => (60,4)
print(y_train.shape) => (90,)
print(y_test.shape) => (60,)
log = LogisticRegression()
log.fit(X_train,y_train)
y_pred = log.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred)) => 0.95
