import pickle
import os

class Model:

    path = 'models/random_forest_model.pkl'
    model = None

    def __init__():
        pass

    def save(self, path=None):
        if path is not None:
            self.path = path

        if not os.path.exists('models'):
            os.mkdir('models')

        if not os.path.exists(self.path):
            os.mknod(self.path)

        # Dump the trained decision tree classifier with Pickle
        # Open the file to save as pkl file
        model_pkl = open(self.path, 'wb')
        pickle.dump(self.model, model_pkl)
        # Close the pickle instances
        model_pkl.close()

        return self.path

    def load(self, path):
        # Load from file
        with open(path, 'rb') as file:
            self.model = pickle.load(file)

        return self.model
