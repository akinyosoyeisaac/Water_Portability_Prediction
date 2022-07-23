from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pickle as pk
import yaml
from argparse import ArgumentParser
from scr.logs import get_logger
import shelve



def config_loader(path:str):
    with open(path) as file:
        config = yaml.safe_load(file)
    return config
    



def train(config):
    
    logger = get_logger('TRAINING STAGE', log_level=config['loglevel'])
    
    logger.info('Loading the train set')
    with shelve.open(config["paths"]["train_set"]) as train_set:
        X_train = train_set["X_test"]
        y_train = train_set["y_test"]
    logger.info('Training set successfully loaded')
    
    logger.info('training the model...')
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy=config["train"]["imputer"])
    
    model_rf = RandomForestClassifier(random_state=config["train"]["seeds"])
    pipe_rf = make_pipeline(imputer, scaler, model_rf)
    pipe_rf.fit(X_train, y_train)
    logger.info('model training successful...')
    
    logger.info('saving the model to ' + config["paths"]["model"] + "...")
    with open(config["paths"]["model"], "wb") as file:
        pk.dump(pipe_rf, file)
    logger.info('model successfully saved')
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", "--p", default="params.yaml", dest="path", type=str, required=True)
    args = parser.parse_args()
    param_path = args.path
    
    config = config_loader(param_path)
    train(config)