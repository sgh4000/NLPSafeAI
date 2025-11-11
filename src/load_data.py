import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd

def load_data_kaggle(file_path):

    # Load the latest version
    df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "infamouscoder/depression-reddit-cleaned",
    file_path,
    # Provide any additional arguments like
    # sql_query or pandas_kwargs. See the
    # documenation for more information:
    # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
    )

    return df