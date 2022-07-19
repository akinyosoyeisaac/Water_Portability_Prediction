import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pickle as pk
import yaml
from argparse import ArgumentParser
from scr.logs import get_logger



def config_loader(path:str):
    with open(path) as file:
        config = yaml.safe_load(file)
    return config
    

def imblearn(X_train, y_train, config):
    imputer = SimpleImputer(strategy=config["train"]["imputer"])
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    imb = SMOTE(random_state=config["train"]["seeds"], k_neighbors=10)
    X_train, y_train = imb.fit_resample(X_train, y_train)
    return X_train, y_train


def train(config):
    
    logger = get_logger('TRAINING STAGE', log_level=config['loglevel'])
    
    logger.info('Loading data...')
    df = pd.read_csv(config["path"]["data"])
    logger.info('Data loaded into memory successfully...')
    
    
    logger.info('Splitting the data into train data and test data...')
    X = df.drop(columns="Potability")
    y = df["Potability"]

    sss = StratifiedShuffleSplit(n_splits=config["train"]["n_splits"], test_size=config["train"]["test_size"], random_state=config["train"]["seeds"])

    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
    logger.info('splitting successfully...')
        
    logger.info('Scaling up the training set for imbalance...')
    X_train, y_train = imblearn(X_train=X_train, y_train=y_train, config=config)
    logger.info('scaling successful...')
    
    logger.info('training the model...')
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy=config["train"]["imputer"])
    
    model_rf = RandomForestClassifier(random_state=config["train"]["seeds"])
    pipe_rf = make_pipeline(imputer, scaler, model_rf)
    pipe_rf.fit(X_train, y_train, config)
    logger.info('model training successful...')
    
    logger.info('saving the model to ' + config["path"]["model"] + "...")
    with open(config["path"]["model"], "wb") as file:
        pk.dump(pipe_rf, file)
    logger.info('model successfully saved')
        
    if __name__ == "__main__":
        parser = ArgumentParser()
        parser.add_argument("--path", "--p", default="params.yaml", dest="path", type=str, required=True)
        args = parser.parse_args()
        param_path = args.path
        
        config = config_loader(param_path)
        train(config)