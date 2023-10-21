import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.base import clone
from tqdm import tqdm


class ModelTuner:
    """
    A class for tuning and evaluating machine learning models.

    Parameters:
    - pipeline (object): Scikit-learn pipeline object representing the machine
      learning model.
    - parameters (list): List of dictionaries containing the hyperparameter
      combinations to tune.
    - X_train (array-like): Training input samples.
    - y_train (array-like): Target values for training.
    - X_valid (array-like): Validation input samples.
    - y_valid (array-like): Target values for validation.
    - eval_metric (callable): Metric to evaluate the model performance.
    - typeOfMLProblem (str, optional): Type of machine learning problem.
      Default is "binary".
    - strategy (str, optional): Multiclass strategy for classification.
      Default is "ovo".

    Attributes:
    - pipeline (object): Scikit-learn pipeline object representing the machine
                         learning model.
    - parameters (list): List of dictionaries containing the hyperparameter
                         combinations to tune.
    - best_model (object): Best performing model based on the chosen
                           evaluation metric.
    - predictions (list): List of predicted values or decision scores for
                          each parameter combination.
    - performance (list): List of performance scores for each
                          parameter combination.
    - best_params (dict): Hyperparameter combination that achieved the
                          best performance.
    - best_performance (float): Best performance score achieved.

    Methods:
    - None

    """

    def __init__(
        self,
        pipeline,
        parameters,
        X_train,
        y_train,
        X_valid,
        y_valid,
        eval_metric,
        typeOfMLProblem="binary",
        strategy="ovo",
    ):
        self.pipeline = pipeline
        self.parameters = parameters
        self.best_model = None

        if typeOfMLProblem == "multi-class":
            self.predictions = []
            self.performance = []

            label_encoder = LabelEncoder()
            label_encoder.fit(y_train)  # Fit label encoder on training data only
            y_train_encoded = label_encoder.transform(y_train)
            y_valid_encoded = label_encoder.transform(y_valid)

            for params in tqdm(parameters):
                pipeline.set_params(**params)

                if strategy == "ovo":
                    clf = OneVsOneClassifier(pipeline)
                elif strategy == "ovr":
                    clf = OneVsRestClassifier(pipeline)
                else:
                    raise ValueError("Invalid multiclass strategy specified.")

                clf.fit(X_train, y_train_encoded)

                # Calculate the decision function or predicted probabilities
                # based on the strategy
                if strategy == "ovo":
                    decision_scores = clf.decision_function(X_valid)
                elif strategy == "ovr":
                    decision_scores = clf.predict_proba(X_valid)

                self.predictions.append(decision_scores)

                if eval_metric.__name__ == "roc_auc_score":
                    # Calculate the ROC AUC separately for each class
                    class_auc = []
                    for class_idx in range(len(label_encoder.classes_)):
                        y_true_class = np.where(y_valid_encoded == class_idx, 1, 0)
                        y_score_class = decision_scores[:, class_idx]
                        class_auc.append(roc_auc_score(y_true_class, y_score_class))

                    # Compute the max of the class-specific ROC AUC
                    max_auc = np.max(class_auc)
                    self.performance.append(max_auc)
                else:
                    self.performance.append(
                        eval_metric(y_valid_encoded, decision_scores)
                    )
            # Get the best parameters based on the highest performance
            self.best_params = self.parameters[np.argmax(self.performance)]

            if strategy == "ovo":
                # Set the best model as One-vs-One classifier with best parameters
                self.best_model = OneVsOneClassifier(
                    pipeline.set_params(**self.best_params)
                )
            elif strategy == "ovr":
                # Set the best model as One-vs-Rest classifier with best parameters
                self.best_model = OneVsRestClassifier(
                    pipeline.set_params(**self.best_params)
                )
            # Fit the best model with the training data
            self.best_model.fit(X_train, y_train_encoded)

        elif typeOfMLProblem == "binary":
            # Generate predictions using pipeline and evaluate performance
            self.predictions = [
                pipeline.set_params(**params)
                .fit(X_train, y_train)
                .predict_proba(X_valid)[:, 1]
                for params in tqdm(parameters)
            ]
            self.performance = [
                eval_metric(y_valid, prediction) for prediction in self.predictions
            ]

            # Get the best parameters based on the highest performance
            self.best_params = self.parameters[np.argmax(self.performance)]

            # Set the best model as pipeline with best parameters
            self.best_model = pipeline.set_params(**self.best_params)
            # Fit the best model with the training data
            self.best_model.fit(X_train, y_train)

        elif typeOfMLProblem == "regression":
            # Generate predictions using pipeline and evaluate performance
            self.predictions = [
                pipeline.set_params(**params).fit(X_train, y_train).predict(X_valid)
                for params in tqdm(parameters)
            ]
            self.performance = [
                eval_metric(y_valid, prediction) for prediction in self.predictions
            ]

            # Get the best parameters based on the highest performance
            self.best_params = self.parameters[np.argmax(self.performance)]

            # Set the best model as pipeline with best parameters
            self.best_model = clone(self.pipeline).set_params(
                **self.best_params
            )  # Change here
            # Fit the best model with the training data
            self.best_model.fit(X_train, y_train)

        # Get the best performance achieved
        self.best_performance = np.max(self.performance)