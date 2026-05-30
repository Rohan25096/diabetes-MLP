import sys 
import os
from dataclasses import dataclass

from sklearn.ensemble import(
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifact","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Splitting train and test input data")
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models={
                'Decision Tree': DecisionTreeClassifier(),
                'Random Forest': RandomForestClassifier(),
                'K Nearest Neighbors': KNeighborsClassifier(),
                'Gradient Boosting': GradientBoostingClassifier(),
                'Ada Boosting': AdaBoostClassifier(),
                'XG Boost': XGBClassifier(),
            }
            
            model_report:dict=evaluate_model(X_train=X_train, y_train=y_train, 
                                             X_test=X_test, y_test=y_test, models=models)
            
            # To get the best model score from the dict.
            best_model_score = max(sorted(model_report.values()))

            # To get the best model name form the dict.
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found.")

            logging.info("Best model found on both trainning and test data")

            save_object(
                        file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model
            )

            predicted_output=best_model.predict(X_test)
            accuracy = accuracy_score(predicted_output,y_test)

            return accuracy

        except Exception as e:
            raise CustomException(e,sys)