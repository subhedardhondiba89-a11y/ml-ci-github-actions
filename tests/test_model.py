from sklearn.datasets import load_iris
from src.model import train_model

def test_model_training():
    model, accuracy = train_model()
    assert model is not None
    assert accuracy >= 0.90

def test_model_predictions():
    model, _ = train_model()
    iris = load_iris()
    predictions = model.predict(iris.data[:5])
    assert len(predictions) == 5
    assert all(p in [0, 1, 2] for p in predictions)
