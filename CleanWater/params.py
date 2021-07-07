### MLFLOW configuration - - - - - - - - - - - - - - - - - - -

MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "[UK] [London] [PKR] CleanWater"

### DATA & MODEL LOCATIONS  - - - - - - - - - - - - - - - - - - -

PATH_TO_LOCAL_MODEL = 'model.joblib'

# AWS_BUCKET_TEST_PATH = "s3://wagon-public-datasets/cleanwater.csv"

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'wagon-data-597-cleanwater' ## --> WIP

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location 
# BUCKET_TRAIN_DATA_PATH = 'XXXX' 
# --> Check trainer.py for the dataset

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'cleanwater'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'

### - - - - - - - - - - - - - - - - - - - - - - - - - - - -