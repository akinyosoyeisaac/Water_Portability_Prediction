from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay, roc_curve
import matplotlib.pyplot as plt
import yaml
import pickle as pk
import json
from scr.logs import get_logger
from argparse import ArgumentParser
import shelve


def config_loader(path:str):
    with open(path) as file:
        config = yaml.safe_load(file)
    return config

def confusion_matrix(X_test, y_test, config, pipe_rf):
    fig, ax = plt.subplots(figsize=(7,7))
    ConfusionMatrixDisplay.from_estimator(pipe_rf, X_test, y_test, normalize='all', labels=[0, 1], display_labels=["Not Potable", "Potable"], ax=ax)
    plt.title("Confusion Matrics".upper(), size=20, weight="bold")
    plt.savefig(config["paths"]["confusion_matrix"])
    
def roc_curve(pipe_rf, X_test, y_test, config):   
    fig, ax = plt.subplots(figsize=(7,7))
    RocCurveDisplay.from_estimator(estimator=pipe_rf, X=X_test, y=y_test, pos_label=1, name="RF", ax=ax)
    plt.title("ROC using Random Forest".upper(), size=20, weight="bold")
    plt.savefig(config["paths"]["roc_curve"])

def evalution(config):
    logger = get_logger('EVALUATION', log_level=config['loglevel'])

    logger.info('Loading the test set')
    with shelve.open(config["paths"]["test_set"]) as test_set:
        X_test = test_set["X_test"]
        y_test = test_set["y_test"]
    logger.info('Test set successfully loaded')

    logger.info('loading save model into memory')
    with open(config["paths"]["model"], "rb") as file:
        pipe_rf = pk.load(file)
    logger.info('model successfully loaded...')

    logger.info('Building classification report')
    metrics = classification_report(y_test, pipe_rf.predict(X_test))
    
    with open(config["paths"]["metrics"], "w") as file:
        json.dump({"metrics": metrics}, file)
    logger.info('Classification report saved')
    
    logger.info('Building confusion matrix')
    confusion_matrix(X_test, y_test, config, pipe_rf)
    logger.info('Saving confusion matrix')
    
    logger.info('Building the roc curve')
    roc_curve(pipe_rf, X_test, y_test, config)
    logger.info('Saving the roc curve')
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", "--p", default="params.yaml", dest="path", type=str, required=True)
    args = parser.parse_args()
    param_path = args.path
    
    config = config_loader(param_path)
    evalution(config)