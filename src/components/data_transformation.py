import os
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from src.loggers import logging
from src.exceptions import CustomException
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from src.utils import save_object
from src.components.data_ingestion import DataIngestion
from src.components.data_ingestion import DataIngestionConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def data_transformer(self):
        try:
            numerical_columns=["writing_score","reading_score"]
            categorical_columns=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy='most_frequent')),
                ("ohe",OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
                ("scaler",StandardScaler())
                ]
            )

            logging.info(f"Numerical Columns: {numerical_columns}")
            logging.info(f"Categorical Column: {categorical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipelines",cat_pipeline,categorical_columns)
                ],remainder="passthrough"
            )

            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
        
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read the Train and Test DataFrame")

            logging.info("Obtaining PreProcessing Object")

            preprocessing_obj=self.data_transformer()
                
            target_column="math_score"
            

            input_feature_train_df=train_df.drop(columns=[target_column],axis=True)
            target_feature_train_df=train_df[target_column]

            input_feature_test_df=test_df.drop(columns=[target_column],axis=True)
            target_feature_test_df=test_df[target_column]

            logging.info("Applying Preprocessing Object on training and testing Dataframe.")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            logging.info("Saved preprocessing object")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)
        



if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modetrainer=ModelTrainer()
    print(modetrainer.initiate_model_trainer(train_arr,test_arr))
            

