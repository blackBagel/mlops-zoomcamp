import pandas as pd
from datetime import datetime
from tests.test_batch import make_testing_data
import batch
import os

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

def test_writing_results_to_s3():
    os.system('python batch.py 2023 1')

    predictions_file = batch.get_output_path(2023, 1)
    df_pred = pd.read_parquet(predictions_file, storage_options=batch.read_parquet_options)
    sum_of_pred = sum(df_pred['predicted_duration'])
    print(f'sum of predicted durations in test is: {sum_of_pred}')



if __name__ == '__main__':
    test_s3_integration()
    test_writing_results_to_s3()