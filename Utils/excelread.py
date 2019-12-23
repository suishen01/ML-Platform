import pandas as pd

class ExcelReader:
    # Class attributes
    path = ''
    features = None

    # Initializer
    def __init__(self, path=None):
        self.path = path

    def read(self, path):
        self.path = path
        self.features = pd.read_excel(self.path)
        print('The features have been successfully imported.')
        return self.features

    def getFeatures(self):
        return self.features

    def setFeatures(self, features):
        self.features = features
