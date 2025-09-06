import os
import sys
from pathlib import Path
from dataclasses import dataclass
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import(
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from src.exceptions import CustomException
from src.loggers import logging
from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=Path('artifacts') / 'models.pkl'

class ModelTrainer:
     def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()



     def initiate_model_trainer(self,train_array,test_array):
         try:
             logging.info("Split training and test data")

             X_train,y_train,X_test,y_test=(
                 train_array[:,:-1],
                 train_array[:,-1],
                 test_array[:,:-1],
                 test_array[:,-1]
             )

             models={
                 "Random Forest": RandomForestRegressor(),
                 "XGB":XGBRegressor(),
                 "AdaBoost":AdaBoostRegressor(),
                 "Gradient Boost":GradientBoostingRegressor(),
                 "Decision Tree":DecisionTreeRegressor(),
                 "Linear Regression": LinearRegression(),
                 "KNN":KNeighborsRegressor(),
                 "CatBoost":CatBoostRegressor(verbose=False)
             }

             model_report:dict=evaluate_models(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,models=models)

             best_model_score=max(sorted(model_report.values()))

             best_model_name=list(model_report.keys())[
                 list(model_report.values()).index(best_model_score)

             ]
             best_model=models[best_model_name]

             logging.info("Best Found Model found in Training and test Dataset")

             save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
             )
             predict=best_model.predict(X_test)
             R2_score=r2_score(y_test,predict)
             return R2_score
             
         except Exception as e:
             raise CustomException(e, sys)
             
    