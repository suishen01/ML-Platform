from sklearn.preprocessing import Imputer

class Cleaner:
    # Class attributes
    data = None
    strategy = 'mean_col'

    # Initializer
    def __init__(self, data=None):
        self.data = data

    def clean(self, strategy='mean_col'):
        self.strategy = strategy

        if strategy == 'mean_col':
            imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
        elif strategy == 'mean_row':
            imputer = Imputer(missing_values='NaN', strategy='mean', axis=1)
        else:
            pass

        imputer.fit(self.data)
        return imputer.transform(self.data.values)
