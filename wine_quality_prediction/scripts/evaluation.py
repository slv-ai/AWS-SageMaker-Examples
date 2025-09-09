"Evaluations script for measuring mean squared error, mean absolute error, and r squared."
import json
import pathlib
import pickle
import tarfile
import numpy as np
import pandas as pd
import xgboost
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.debug("Starting evaluation.")
    model_path = f"/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    
    logger.debug("Loading xgboost model.")
    model = pickle.load(open("xgboost-model", "rb"))

    logger.debug("Reading test data.")
    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path, header=None)
    
    y_test = df.iloc[:, 0].to_numpy()
    df.drop(df.columns[0], axis=1, inplace=True)
    X_test = xgboost.DMatrix(df.values)
    
    logger.info("Performing predictions against test data.")
    predictions = model.predict(X_test)


    # View a plot of the model predictions vs the actuals
    logger.info("Plotting test predictions vs actuals.")
    plt.figure(figsize=(10,10))
    plt.scatter(y_test, predictions, c='crimson')   
    p1 = max(max(predictions), max(y_test))
    p2 = min(min(predictions), min(y_test))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.show()
    print("plot", plt.show())



    # Evaluation Metrics
    
    logger.info("Calculating evaluation metrics.")
    mse = mean_squared_error(y_test, predictions)
    std = np.std(y_test - predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    report_dict = {
        "regression_metrics": {
            "mse": {
                "value": mse,
                "standard_deviation": std
            },
            "mae": {"value": mae,},

            "r_squared": {"value": r2}
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("Writing out evaluation report with mse: %f and r2: %f", mse, r2)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))


