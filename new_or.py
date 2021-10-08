from oneNeuron.perceptron import Perceptron
#from utils.model import Perceptron
from utils.all_utils import  prepare_data, save_model, save_plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib # FOR SAVING MY MODEL AS A BINARY FILE
from matplotlib.colors import ListedColormap
import logging
import os

logging_str= "[%(asctime)s : %(levelname)s : %(module)s] %(message)s " 
log_dir= "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"),level=logging.INFO, format=logging_str, filemode="a")


def main(data, eta, epochs, filename,plotFileName):

    df = pd.DataFrame(data)

    logging.info(f"this is a dataframe{df}")

    X,y = prepare_data(df)

    model = Perceptron(eta=ETA, epochs=EPOCHS)
    model.fit(X, y)

    _ = model.total_loss()

    save_model(model,filename=filename)
    save_plot(df,plotFileName ,model)

if __name__ == '__main__':
    OR = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,1,1,1],
    }

    ETA = 0.3 # 0 and 1
    EPOCHS = 10
    try:
        logging.info(">>>>>>> Starting >>>>>>> ")
        main(OR, ETA, EPOCHS,filename="or.model",plotFileName="or.png")
        logging.info("<<<<<<<<<< Training done successfully <<<<<<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e