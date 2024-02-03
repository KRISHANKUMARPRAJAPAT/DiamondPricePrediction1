from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline     # creating the pipelines
from sklearn.compose import ColumnTransformer   # combining two pipelines

import sys, os 
from dataclasses import dataclass
import pandas as pd 
import numpy as np 

from kpr.exception import CustomException
from kpr.logger import logging

from kpr.utils import save_object

# Data Transformation config
@dataclass
class DataTransformationconfig:
    preprocessor_ob_file_path = os.path.join('artifacts', 'preprocessor.pkl')


# Data Transformationconfig class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()


    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            # Define which columns should be oridinal-encoded and which should be scaled
            categorical_cols = ['cut','color','clarity']
            numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

            # define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']   # [fair --> Ideal (increasing order)]
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']            # [D--> J (Increasing order)]
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']   # [I1 --> IF (increasing order)]

            logging.info('Pipeline initiated')

            ### Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ('scaler',StandardScaler())
                ]

            )

            # combining the pipelines
            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])

            return preprocessor
        
            logging.info('Pipeline Completed')
        
        except Exception as e :
            logging.info('Error in Data Transformation')
            raise CustomException(e,sys)




    def initiate_data_transformation(self,train_path, test_path):
        try:
            # Reading train and test data 
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'price'
            drop_columns = [target_column_name,'id']   # we don't need id column

            # Features into independent and dependent features

            input_feature_train_df = train_df.drop(columns= drop_columns, axis= 1)  # independent feature for training df
            target_feature_train_df = train_df[target_column_name]                  # dependent feature for training df


            input_feature_test_df = test_df.drop(columns=drop_columns, axis= 1)  # for testing df
            target_feature_test_df = test_df[target_column_name]         # for test df

            # Apply the transformation
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")  

            #conbining target feature and input feature into train test
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_obj


            )

            logging.info('Preprocessor pickle is created and saved')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )
        
        except Exception as e:
            logging.info("Exception occured in the initiate_data_transformation")

            raise CustomException(e,sys)

