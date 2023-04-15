# yascikit-learn
Yet another scikit-learn

## Installation
```
pip install yascikit-learn
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
snb = SelectiveNB().fit(X, y)
snb.predict(X)
```
#### Universal Set Naive Bayes
```python
from yasklearn.naive_bayes import UniversalSetNB
from sklearn import datasets

dataset = datasets.load_iris()
X = dataset.data
y = dataset.target
unb = UniversalSetNB().fit(X, y)
unb.predict(X)
```

### FTRLProximal
```python
from yasklearn.ftrl_proximal import FTRLProximalClassifier
from sklearn import datasets

dataset = datasets.load_iris()
X = dataset.data
y = dataset.target
ftrlc = FTRLProximalClassifier().fit(X, y)
ftrlc.predict(X)
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
from sklearn.datasets import fetch_20newsgroups

newsgroups = fetch_20newsgroups(subset='train')
X = list(map(lambda x: x.split(), newsgroups.data))
plsv = PLSV(n_components=20, n_dimension=2, random_state=1)
plsv.fit_transform(X)
```

### Clustering
#### XMeans
```python
from yasklearn.cluster import XMeans
from sklearn import datasets

dataset = datasets.load_iris()
X = dataset.data
xm = XMeans(n_clusters=3, random_state=1)
xm.fit_predict(X)
```

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
