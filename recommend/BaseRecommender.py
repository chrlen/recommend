import abc
import dill
import time

class Recommender(abc.ABC):
    def __init__(self, name, **kwargs):
        abc.ABC.__init__(self)
        self.name = name + "_" + str(kwargs).\
            replace(' ', '').\
            replace('\'', '').\
            replace(',', '_').\
            replace('}', '').\
            replace('{', '')

    @abc.abstractmethod
    def predict(self, data):
        """Takes coordinates of values as predicts if no value is known"""


    @abc.abstractmethod
    def fit(self, data):
        """fits the Recommender to provided data"""

    def predictAndSave(cls, path='predictions/'):
        """predict"""

    def save(self, model_file_name):
        with open(model_file_name, "wb") as f:
            dill.dump(self, f)



