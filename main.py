from model import Model
from torch import nn, optim

def get_data():
    """
    Simple function to get the data
    """
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pd.read_csv(url, names=names)
    array = dataset.values 
    
    return array

def get_XY():
    """
    Function to get X (inputs)
    and Y (outuputs).
    """
    X, Y = array[:, :4], array[:, 4]
    return X, Y

def get_loss():
    loss_function = nn.CrossEntropyLoss()
    # Optionally, try Adagrad and Adam algos here
    opt = optime.SGD(model.parameters, )

