import pickle
import pandas as pd  
from warnings import filterwarnings
filterwarnings("ignore")

with open('./model/preprocessing.pkl', 'rb') as f:
    processing = pickle.load(f)

    cols_order = processing["cols_order"]
    num_cols = processing["num_cols"]
    uppers = processing["uppers"]
    lowers = processing["lowers"]
    scaler = processing["scaler"]
    cols_to_enocde = processing["cols_to_enocde"]
    KBinsDiscretizer = processing["KBinsDiscretizer"]
    model = processing["model"]


def get_data(response):
    data_dict = dict(response)
    cols, vals = list(data_dict.keys()), list(data_dict.values())
    data = []
    for val in vals:
        try:
            data.append(float(val))
        except:
            data.append(val)
    data = [data]
    d = pd.DataFrame(data, columns=cols)
    return d

def preprocess(d):
    # re-arrange columns order
    x = d[cols_order]
    
    # Handle Outliers
    for col in num_cols:
        upper = uppers[col]
        lower = lowers[col]
        upper_outliers = x[x[col] >= upper][col].values
        lower_outliers = x[x[col] <= lower][col].values
        x[col].replace(upper_outliers, upper, inplace=True)
        x[col].replace(lower_outliers, lower, inplace=True)
    
    # Normalization
    x[num_cols] = scaler.transform(x[num_cols])
    
    # Encoding
    x = KBinsDiscretizer.transform(x)
    
    return x


def Model():
    return model

def predict(model, x):
    pred = model.predict(x)
    return pred

