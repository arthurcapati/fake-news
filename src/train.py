from sklearn.model_selection import train_test_split, GridSearchCV
from src.load_data import DataProcessor
import mlflow
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_sklearn_models(model, params):
    mlflow.sklearn.autolog()
    random_state = params.get("random_state", None)
    data = DataProcessor()
    mlflow.log_input(data.dataset, "training")
    X_train, X_test, y_train, y_test = train_test_split(data.X, data.y, test_size=0.2, random_state=random_state)

    if model.__name__ == "MultinomialNB":
        params.pop("random_state", None)
    model = model(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_score = roc_auc_score(y_test, y_pred_proba)

    if model.__class__.__name__ == "MultinomialNB":
        params["random_state"] = random_state
    mlflow.log_params(params)
    mlflow.log_metrics({"test_accuracy": accuracy, "test_precision": precision, "test_recall": recall, "test_f1": f1, "test_roc_auc_score": roc_score})

def grid_search(model, params, grid, cv=5):
    mlflow.autolog()
    random_state = params.get("random_state", None)
    data = DataProcessor()
    mlflow.log_input(data.dataset, "training")
    X_train, X_test, y_train, y_test = train_test_split(data.X, data.y, test_size=0.2, random_state=random_state)

    if model.__name__ == "MultinomialNB":
        params.pop("random_state", None)
    model = model(**params)
    grid_search = GridSearchCV(model, grid, cv=cv, scoring="roc_auc")
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    model = grid_search.best_estimator_
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_score = roc_auc_score(y_test, y_pred_proba)

    if model.__class__.__name__ == "MultinomialNB":
        best_params["random_state"] = random_state
    mlflow.log_params(best_params)
    mlflow.log_metrics({"test_accuracy": accuracy, "test_precision": precision, "test_recall": recall, "test_f1": f1, "test_roc_auc_score": roc_score})
    
    
    pass