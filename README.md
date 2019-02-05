# yascikit-learn
Yet another scikit-learn

## Installation
```
pip install git+https://github.com/ikegami-yukino/yascikit-learn
```

## USAGE
### Naive Bayes
#### Negation Naive Bayes
```python
from yasklearn.naive_bayes import NegationNB
from sklearn import datasets

dataset = datasets.load_iris()
X = dataset.data
y = dataset.target
nnb = NegationNB().fit(X, y)
nnb.predict(X)
```
#### Selective Naive Bayes
```python
from yasklearn.naive_bayes import SelectiveNB
from sklearn import datasets

dataset = datasets.load_iris()
X = dataset.data
y = dataset.target
nnb = SelectiveNB().fit(X, y)
nnb.predict(X)
```
#### Universal Set Naive Bayes
```python
from yasklearn.naive_bayes import UniversalSetNB
from sklearn import datasets

dataset = datasets.load_iris()
X = dataset.data
y = dataset.target
nnb = UniversalSetNB().fit(X, y)
nnb.predict(X)
```

### Topic modeling
#### PLSA
```python
from yasklearn.decomposition import PLSA
from sklearn import datasets

dataset = datasets.load_iris()
X = dataset.data
plsa = PLSA(n_components=3, random_state=1).fit(X)
plsa.predict(X)
```
#### PLSV
Note that PLSV has not implemented predict method.
```python
from yasklearn.decomposition import PLSV
from sklearn import datasets

dataset = datasets.load_iris()
X = dataset.data
plsv = PLSV(n_components=3, n_dimension=2, random_state=1)
plsv.fit_transform(X)
```

### Clustering
Note that KMedoids and XMedoids have not implemented predict method.
#### KMedoids
```python
from yasklearn.cluster import KMedoids
from sklearn import datasets

dataset = datasets.load_iris()
X = dataset.data
km = KMedoids(n_clusters=3, random_state=1)
km.fit_predict(X)
```
#### XMedoids
```python
from yasklearn.cluster import XMedoids
from sklearn import datasets

dataset = datasets.load_iris()
X = dataset.data
xm = XMedoids(n_clusters=3, random_state=1)
xm.fit_predict(X)
```

### Utility
```python
from yasklearn.model_selection import train_dev_test_split
import numpy as np

X = np.arange(10).reshape((5, 2))
y = range(5)
X_train, X_dev, X_test, y_train, y_dev, y_test = train_dev_test_split(
    X, y, dev_size=0.33, random_state=1)
```
