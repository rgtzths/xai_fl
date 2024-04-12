import numpy as np
from alibi.explainers import PermutationImportance, plot_permutation_importance


def results(X_train, y_train, X_test, y_test, model):
    features = X_train.columns.tolist()
    target = y_train.name

    # To numpy
    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values
    y_test = y_test.values

    def predict_fn(X):
        return np.argmax(model.predict(X), axis=1)
    
    explainer = PermutationImportance(predictor=predict_fn, score_fns='accuracy', feature_names=features)
    explanations = explainer.explain(X=X_test, y=y_test, kind='difference')

    # plot_permutation_importance(explanations)

    f_names = explanations.data['feature_names']
    f_importance = explanations.data['feature_importance'][0]
    importances = {f_names[i]: f_importance[i]['mean'] for i in range(len(f_names))}

    return importances
