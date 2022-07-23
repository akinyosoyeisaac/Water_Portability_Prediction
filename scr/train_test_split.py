import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import yaml
from argparse import ArgumentParser
from scr.logs import get_logger
import shelve




def imblearn(X_train, y_train):
    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    resample = SMOTE(random_state=234, k_neighbors=10)
    X_train, y_train = resample.fit_resample(X_train, y_train)
    return X_train, y_train

def config_loader(path:str):
    with open(path) as file:
        config = yaml.safe_load(file)
    return config

def train_test_split(config):    
    
    logger = get_logger('TRAIN TEST SPLIT', log_level=config['loglevel'])
    
    logger.info('Loading data...')
    df = pd.read_csv(config["paths"]["raw_data"])
    logger.info('Data loaded into memory successfully...')

    X = df.drop(columns="Potability")
    y = df["Potability"]

    logger.info('Split the data into train test split')
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=234)

    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
    logger.info('the data has been successfully splitted into train and test data...')
    
    logger.info('Handling imblance in the train data')
    X_train, y_train = imblearn(X_train=X_train, y_train=y_train)
    logger.info('The minority class has been oversampled using SMOTE...')
    
    logger.info('The X_test is been save for the for the purpose of test our model in production environment')
    test_df = X_test

    test_df.to_csv(config["paths"]["test_data"], index=False)
    logger.info('the X_test set has been saved in ' + config["paths"]["test_data"])
    
    
    logger.info('The test_set is been saved for the purpose of evaluation')
    with shelve.open(config["paths"]["test_set"]) as test_set:
        test_set["X_test"] = X_test
        test_set["y_test"] = y_test
    logger.info('The test set has been saved in ' + config["paths"]["test_set"])
        
    logger.info('The train set is been saved for the purpose of training')
    with shelve.open(config["paths"]["train_set"]) as train_set:
        train_set["X_test"] = X_train
        train_set["y_test"] = y_train
    logger.info('The train set has been saved in ' + config["paths"]["train_set"])
        
if __name__ == "__main__":
        parser = ArgumentParser()
        parser.add_argument("--path", "--p", default="params.yaml", dest="path", type=str, required=True)
        args = parser.parse_args()
        param_path = args.path
        config = config_loader(param_path)
        train_test_split(config)