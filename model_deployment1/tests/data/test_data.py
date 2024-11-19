import pytest

@pytest.mark.data
def test_dataset(df):
    expected_column_list=["userId","movieId","rating","timestamp"]
    column_list=df.columns.tolist()
    assert column_list==expected_column_list