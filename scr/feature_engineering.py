import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import yaml
from argparse import ArgumentParser
from scr.logs import get_logger
import shelve


def config_loader(path:str):
    with open(path) as file:
        config = yaml.safe_load(file)
    return config

def feature_engineering(config):    
    
    logger = get_logger('FEATURE ENGINEERING', log_level=config['loglevel'])
    
    logger.info('Loading data...')
    df = pd.read_csv(config["paths"]["raw_data"])
    logger.info('Data loaded into memory successfully...')
    
    logger.info('Feature Engineering the data...')
    ph_mean = df[df['Potability'] == 0]['ph'].mean(skipna=True)
    df.loc[(df['Potability'] == 0) & (df['ph'].isna()), 'ph'] = ph_mean

    ph_mean_1 = df[df['Potability'] == 1]['ph'].mean(skipna=True)
    df.loc[(df['Potability'] == 1) & (df['ph'].isna()), 'ph'] = ph_mean_1

    sulf_mean = df[df['Potability'] == 0]['Sulfate'].mean(skipna=True)
    df.loc[(df['Potability'] == 0) & (df['Sulfate'].isna()), 'Sulfate'] = sulf_mean

    sulf_mean_1 = df[df['Potability'] == 1]['Sulfate'].mean(skipna=True)
    df.loc[(df['Potability'] == 1) & (df['Sulfate'].isna()), 'Sulfate'] = sulf_mean_1

    traih_mean = df[df['Potability'] == 0]['Trihalomethanes'].mean(skipna=True)
    df.loc[(df['Potability'] == 0) & (df['Trihalomethanes'].isna()), 'Trihalomethanes'] = traih_mean

    trah_mean_1 = df[df['Potability'] == 1]['Trihalomethanes'].mean(skipna=True)
    df.loc[(df['Potability'] == 1) & (df['Trihalomethanes'].isna()), 'Trihalomethanes'] = trah_mean_1
    logger.info('Feature Engineering sucessful...')
    
    logger.info('Saving the featured engineered data...')
    df.to_csv(config["paths"]["processed_data"], index=False)
    logger.info('saving processed data sucessfully...')

if __name__ == "__main__":
        parser = ArgumentParser()
        parser.add_argument("--path", "--p", default="params.yaml", dest="path", type=str, required=True)
        args = parser.parse_args()
        param_path = args.path
        config = config_loader(param_path)
        feature_engineering(config)