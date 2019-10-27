from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

params = {'n_neighbors': list(range(3, 10))}
classifier = KNeighborsClassifier()
clf = GridSearchCV(classifier, params, cv=10)
clf.fit(X_train_codes, y_train_codes)