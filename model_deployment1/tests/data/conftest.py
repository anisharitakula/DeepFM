import pytest
import pandas as pd

def pytest_addoption(parser):

    parser.addoption("--dataset1-loc",action="store",default=None,help="Dataset location.")


@pytest.fixture
def df(request):
    data_loc=request.config.getoption("--dataset1-loc")
    return pd.read_csv(data_loc)