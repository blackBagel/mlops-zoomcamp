import pandas as pd
from datetime import datetime
from tests.test_batch import make_testing_data
import batch
import boto3

def test_s3_integration():
    df = make_testing_data()

    input_file = batch.get_input_path(2023, 1)
    
    df.to_parquet(
        input_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=batch.read_parquet_options
    )

if __name__ == '__main__':
    test_s3_integration()