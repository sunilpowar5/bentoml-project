import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

## any number of models can be get for like data transformation like standard scaler, ct etc...

iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

svc = bentoml.Service("Iris_Classifier",runners=[iris_clf_runner])

@svc.api(input=NumpyNdarray(),output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    result = iris_clf_runner.predict.run(input_series)
    return result