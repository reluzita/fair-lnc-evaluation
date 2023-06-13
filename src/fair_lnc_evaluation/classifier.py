import pandas as pd

class Classifier:
    def __init__(self, classifier, name):
        """
        Parameters
        ----------
        classifier : sklearn classifier
            The classifier to use.
        name : str
            The name of the classifier (used for mlflow logging).
        """
        self.classifier = classifier
        self.name = name

    def fit_predict(self, X_train, y_train, X_test):
        """
        Fit the classifier on the training data and predict the probabilities for the test data.

        Parameters
        ----------
        X_train : pandas.DataFrame
            The training features.
        y_train : pandas.Series
            The training labels.
        X_test : pandas.DataFrame
            The test features.

        Returns
        -------
        pandas.Series
            The predicted probabilities for the test data.
        """
        model = self.classifier.fit(X_train, y_train)
        return pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)