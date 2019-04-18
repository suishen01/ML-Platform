from sklearn.model_selection import train_test_split

class Spliter:

    # Class attributes
    X = None
    Y = None

    # Initializer
    def __init__(self, data=None):
        self.data = data

    def split(self, X, Y=None, test_size=0.25, random_state=42):
        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y
            # Split the data into training and testing sets
            train_X, test_X, train_Y, test_Y = train_test_split(self.X, self.Y, test_size = test_size, random_state = random_state)
            return train_X, test_X, train_Y, test_Y
