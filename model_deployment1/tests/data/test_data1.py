import pandas as pd
import pytest


def test_dummy(df):
    assert isinstance(df,pd.DataFrame)