""" Feature engineers the wine quality dataset."""
import numpy as np
import pandas as pd
import glob as gb
import logging

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# Because this is a headerless CSV file, specify the feature and label column names here.
feature_columns_names = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]
label_column = "quality" 

# Specify feature and label data types
feature_columns_dtype = {
    "fixed acidity": np.float64,
    "volatile acidity": np.float64,
    "citric acid": np.float64,
    "residual sugar": np.float64,
    "chlorides": np.float64,
    "free sulfur dioxide": np.float64,
    "total sulfur dioxide": np.float64,
    "density": np.float64,
    "pH": np.float64,
    "sulphates": np.float64,
    "alcohol": np.float64,
    
}
label_column_dtype = {"quality": np.float64}

def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

if __name__ == "__main__":
    logger.info("Starting preprocessing.")

    base_dir = "/opt/ml/processing"
    all_files = gb.glob(f"{base_dir}/input" + "/*.csv")

    li = []
    df = pd.DataFrame()

    # Read in all CSV files in the training input data directory
    for filename in all_files:
        logger.info("Reading data from %s.", filename)
        frames = pd.read_csv(
            filename,
            header=None, 
            names=feature_columns_names + [label_column],
            dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype))
        li.append(frames)

    df = pd.concat(li)

    
    

    
    print("Dataset columns and datatypes: \n", df.dtypes, "\n")
    print("Shape of our dataset:", df.shape, "\n")
    print("Sample Dataset: \n", df.head(), "\n")
    print("Describe Dataset: \n", df.describe().T, "\n")



    # Transform features
    numeric_features = list(feature_columns_names)
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
        ]
    )

    # Split dataset into x and y (features and labels). Applying Transforms.
    logger.info("Applying transforms.")
    y = df.pop("quality")
    X_pre = preprocess.fit_transform(df)
    y_pre = y.to_numpy().reshape(len(y), 1)
        
    X = np.concatenate((y_pre, X_pre), axis=1)

    # Split our data-set into three different files for training, validation, and testing
    logger.info("Splitting %d rows of data into train, validation, test datasets.", len(X))
    np.random.shuffle(X)
    train, validation, test = np.split(X, [int(.7*len(X)), int(.9*len(X))])

    # Write files to Sagemaker directory
    logger.info("Writing out datasets to %s.", base_dir)
    pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)
    pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
