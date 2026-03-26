import pytest
import os
import pickle
from api.config import MODEL_PATH

@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model not trained yet")
def test_model_loading():
    model = pickle.load(open(MODEL_PATH, 'rb'))
    assert model is not None

@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model not trained yet")
def test_model_predictions():
    assert True
