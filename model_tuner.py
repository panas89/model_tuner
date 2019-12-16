from sklearn.pipeline import Pipeline
import numpy as np
from tqdm import tqdm

class ModelTuner:
    def __init__(self,pipeline,parameters,X_train,y_train,X_valid,y_valid,eval_metric):
        self.pipeline = pipeline
        self.parameters = parameters
        self.predictions = [pipeline.set_params(**params).fit(X_train, y_train)\
                                    .predict_proba(X_valid)[:,1] for params in tqdm(parameters)]
        self.performance = [eval_metric(y_valid,prediction) for prediction in self.predictions]
        self.best_params = self.parameters[np.argmax(self.performance)]
        self.best_model = pipeline.set_params(**self.best_params).fit(X_train, y_train)
        self.best_performance = np.max(self.performance)