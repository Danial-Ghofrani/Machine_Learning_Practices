import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sympy.core.benchmarks.bench_numbers import timeit_Integer_mul_Rational
import time

def time_it(function_name):
    def inner(*args, **kwargs):
        start = time.time()
        result = function_name(*args, **kwargs)
        end = time.time()
        return end-start, result
    return inner
def create_data(sample, features, std, train_Size):
    X, y = make_blobs(
        n_samples=samples,
        n_features=features,
        cluster_std=std,
        random_state=32)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=32)
    return x_train, x_test, y_train, y_test

@time_it
def train_model(model, x_train, y_train):
    model.fit(x_train, y_train)
    return model

@time_it
def evaluate_model(model, x_test, y_test, scoring = "accuracy"):
    y_pred = model.predict(x_test)

    match scoring:
        case "accuracy":
            score = accuracy_score(y_test, y_pred)
        case "precision":
            score = precision_score(y_test, y_pred)
        case "recall":
            score = recall_score(y_test, y_pred)
        case "f1-score":
            score = f1_score(y_test, y_pred)
    return score


result_list = []

for samples in [100,500,1000,5000,10000,50000,100000]:
    for features in [2,4,10]:
            for std in [1, 1.5, 3, 5, 7, 12]:
                x_train, x_test, y_train, y_test = create_data(samples, features, std, 0.8)

                model = KNeighborsClassifier(n_neighbors=5)

                train_time, model = train_model(model, x_train, y_train)
                test_time, score = evaluate_model(model, x_test, y_test, scoring="accuracy")


                print(f"n={samples:6}, features={features:3}, std={std:3}, accuracy={score:10}, train_time={train_time}, test_time={test_time}")
                result_list.append(
                    {"n":samples,
                     "features":features,
                     "std":std,
                     "accuracy":score,
                     "train_time":train_time,
                     "test_time":test_time})


df = pd.DataFrame(result_list)
df.to_excel("Dataset_features_comparison_for_KNN.xlsx")
