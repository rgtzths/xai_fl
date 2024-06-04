import numpy as np
from alibi.explainers import PartialDependenceVariance, plot_pd_variance


def results(X_train, y_train, X_test, y_test, model):
    features = X_train.columns.tolist()
    target = y_train.columns.tolist()[0]

    # To numpy
    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values
    y_test = y_test.values

    def predict_fn(X):
        return np.argmax(model.predict(X, verbose=0), axis=1)
    
    explainer = PartialDependenceVariance(predictor=predict_fn, feature_names=features, target_names=[target])
    explanations = explainer.explain(X=X_test, method='importance')

    # plot_pd_variance(exp=explanations)

    f_names = explanations.data['feature_names']
    f_importance = explanations.data['feature_importance'][0].tolist()
    importances = {f_names[i]: f_importance[i] for i in range(len(f_names))}

    return importances
