from shap import KernelExplainer, kmeans
import pandas as pd
import numpy as np


def results(X_train, y_train, X_test, y_test, model):
    features = X_test.columns
    preds = model.predict(X_test)
    background = kmeans(X_train, 100)

    explainer = KernelExplainer(model.predict, background)
    shap_values = explainer.shap_values(X_test, nsamples=64)
    
    results = []
    for i in range(preds.shape[1]):
        class_shape_values = shap_values[i]
        shap_values_mean = np.mean(np.abs(class_shape_values), axis=0)
        results.append(shap_values_mean)

    results = np.array(results)

    df = pd.DataFrame(columns=(["feature"] + [f"y{i}" for i in range(preds.shape[1])]))
    for i in range(len(features)):
        df.loc[i] = [features[i]] + [results[j][i] for j in range(preds.shape[1])]
    return df
