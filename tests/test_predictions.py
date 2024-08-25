"""
Note: These tests will fail if you have not first trained the model.
"""

import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from model.predict import make_prediction

def test_make_prediction(sample_input_data):
    # Given
    expected_no_predictions = 2000

    # When
    result = make_prediction(input_data=sample_input_data[0])

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, np.ndarray)
    assert isinstance(predictions[0], np.int64)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions

    _predictions = list(predictions)
    y_true = sample_input_data[1]

    accuracy = accuracy_score(y_true, _predictions)
    assert accuracy > 0.8
    auc = roc_auc_score(y_true, _predictions)
    assert auc > 0.7
    f1score = f1_score(y_true, _predictions)
    assert f1score > 0.6
