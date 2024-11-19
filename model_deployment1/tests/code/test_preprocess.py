import pytest
from preprocess.preprocessing import process_data

@pytest.mark.code
def test_code(sample_data):
    df1,df2=sample_data
    train_data,test_data=process_data(df1,df2)

    assert train_data.shape[0]==2 and test_data.shape[0]==1

    