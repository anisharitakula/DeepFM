import pytest
import pandas as pd

@pytest.fixture
def sample_data():
    df1=pd.DataFrame({'userId':[1,2,1],'movieId':[4,5,6],'rating':[1.0,3.0,5.0]})
    df2=pd.DataFrame({'movieId':[4,5,6],'title':["Toy story","Titanic","Oldboy"],'genres':["Kids|Comedy","Romance|Tragedy","Thriller|Suspense"]})
    return df1,df2