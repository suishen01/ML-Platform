import pandas as pd

class SQL:
    # Class attributes
    data = None
    # Initializer
    def __init__(self, data=None):
        self.data = data

    def setIndex(self, index):
        self.data = self.data.set_index(index, drop=False)

    def getKeyList(self, key):
        return self.data[key].unique().tolist()

    def select(self, index, where=None, condition='equal'):
        if where == None:
            return self.data[index]
        else:
            if condition == 'equal':
                return self.data[self.data[index] == where]

    def head(self, head):
        return self.data.head(head)

    def sort(self, value_list, ascending=True):
        self.data = self.data.sort_values(value_list, ascending=ascending)
