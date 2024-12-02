import sklearn
import bentoml

from sklearn.svm import SVC
from sklearn.datasets import load_iris

def load_dataset():
    iris_data = load_iris()
    X,y = iris_data.data, iris_data.target
    return X,y

def train_model(x_train,y_train):
    svc_clf = SVC(gamma='scale')
    svc_clf.fit(x_train,y_train)
    return svc_clf

## save model

def save_model(model):
    model = bentoml.sklearn.save_model("iris_clf", model)
    return model

if __name__ == '__main__':
    
    X,y = load_dataset()

    model = train_model(X,y)

    saved_model = save_model(model)

    print(f"saved_model:{saved_model}")

    ## iris_clf:ediesgfqtc3idtti -- for inferencing