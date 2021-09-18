from utils.model import Perceptron
from utils.all_utils import  prepare_data, save_model, save_plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib # FOR SAVING MY MODEL AS A BINARY FILE
from matplotlib.colors import ListedColormap

def main(data, eta, epochs, filename,plotFileName):

    df = pd.DataFrame(data)

    print(df)

    X,y = prepare_data(df)

    model = Perceptron(eta=ETA, epochs=EPOCHS)
    model.fit(X, y)
    _ = model.total_loss()
    save_model(model,filename=filename)
    save_plot(df,plotFileName ,model)

if __name__ == '__main__':
    AND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,0,0,1],
    }
    ETA = 0.3 # 0 and 1
    EPOCHS = 10
    main(AND, ETA, EPOCHS,filename="and.model",plotFileName="and.png")
