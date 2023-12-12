from fanova import fANOVA
import numpy as np
import pandas as pd

SEEDS = [42, 13, 21]

def results(X_train, y_train, X_test, y_test, model):
    step = 0
    features = X_test.columns
    preds = model.predict(X_test)
    preds = preds.astype('float64') 
    total = len(SEEDS) * len(features) * (preds.shape[1])
    
    results = []
    for seed in SEEDS:
        results.append([])
        for i in range(preds.shape[1]):
            results[-1].append([])
            y = preds[:, i]
            f = fANOVA(X=X_test, Y=y, seed=seed)
            for j in range(len(features)):
                res = f.quantify_importance((j,))
                results[-1][-1].append(round(res[(j,)]['individual importance']*100,2))
                step += 1
                print(f"Progress: {step}/{total} ({round(step/total*100,2)}%)", end="\r", flush=True)

    results = np.array(results)
    print(f"results shape before average: {results.shape}")
    # average over seeds
    results = np.mean(results, axis=0)
    print(f"results shape after average: {results.shape}")

    df = pd.DataFrame(columns=(["feature"] + [f"y{i}" for i in range(preds.shape[1])]))
    for i in range(len(features)):
        df.loc[i] = [features[i]] + [results[j][i] for j in range(preds.shape[1])]
    return df