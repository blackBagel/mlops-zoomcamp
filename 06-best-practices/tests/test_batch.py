import batch
import pandas as pd
from datetime import datetime

categorical = ['PULocationID', 'DOLocationID']


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def make_testing_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    return pd.DataFrame(data, columns=columns).copy()

def test_prepare_data():
    test_df = make_testing_data()

    expected_data = [
        ('-1', '-1', dt(1, 1), dt(1, 10), 9.0),
        ('1', '1', dt(1, 2), dt(1, 10), 8.0),  
    ]
    expected_columns = list(test_df.columns) + ['duration']
    expected_df = pd.DataFrame(expected_data, columns=expected_columns)


    actual_df = batch.prepare_data(test_df, categorical=categorical)

    result_df = actual_df.compare(expected_df)

    assert result_df.empty
    

    
