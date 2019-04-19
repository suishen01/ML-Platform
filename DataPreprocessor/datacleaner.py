from sklearn.preprocessing import Imputer
import pandas as pd

class Cleaner:
    # Class attributes
    data = None
    strategy = 'mean_col'

    # Initializer
    def __init__(self, data=None):
        self.data = data

    def clean(self, strategy='mean_col', type='array'):
        self.strategy = strategy

        if strategy == 'mean_col':
            imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
        elif strategy == 'mean_row':
            imputer = Imputer(missing_values='NaN', strategy='mean', axis=1)
        else:
            pass

        imputer.fit(self.data)
        data_array = imputer.transform(self.data)

        count = 0

        self.data = pd.DataFrame(data_array, columns=list(self.data))

        if type == 'df':
            return self.data
        elif type == 'array':
            return data_array
