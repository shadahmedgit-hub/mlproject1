import os
import sys

from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            logging.info("Train Test Split completed")
            models = {"Linear Regression":LinearRegression(),
                      "Decision Tree":DecisionTreeRegressor(),
                      "Random Forest":RandomForestRegressor(),
                      "Gradient Boosting":GradientBoostingRegressor(),
                      "AdaBoost":AdaBoostRegressor()
                      }

            params = {

                    "Linear Regression": {},

                    "Decision Tree": {
                        "criterion": ["squared_error", "friedman_mse"],
                        "max_depth": [5, 10, 15, None]
                    },

                    "Random Forest": {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [5, 10, None]
                    },

                    "Gradient Boosting": {
                        "learning_rate": [0.01, 0.05, 0.1],
                        "n_estimators": [50, 100, 200],
                        "max_depth": [3, 5]
                    },

                    "AdaBoost": {
                        "learning_rate": [0.01, 0.1, 1.0],
                        "n_estimators": [50, 100, 200]
                    }
                }

            model_report:dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param = params
            )

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise Exception("No best model found")
            
            logging.info(f"Best model found: {best_model_name}")

            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)
            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test,predicted)

            return r2_square
        except Exception as e:
            raise CustomException(e, sys)
