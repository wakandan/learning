"""
Bagging = ensemble with random sampling with replacement
Pasting = ensemble with random sampling without replacement
"""

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=100,
    bootstrap=True,  # True => bagging, False => pasting
    n_jobs=-1,  # use all cores
    oob_score=True
)

X, y = make_moons(n_samples=int(1e6), noise=0.3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y)
bag_clf.fit(X_train, y_train)
print("out of bag score", bag_clf.oob_score_)
y_pred = bag_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))  # => 0.911788
