import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pathlib


def preprocessing(kpi_file, distances_file, met_forecast_file, met_real_file, output_path, prediction_interval=5, look_back_interval=10):
    output = pathlib.Path(output_path)
    output.mkdir(parents=True, exist_ok=True)

    identifiers = ["site_id", "mlid", "datetime"]
    df = pd.read_csv(kpi_file, sep="\t", index_col=0)

    if "datetime" in df:
        df["datetime"] = pd.to_datetime(df["datetime"])

    df_labels = df.loc[:, identifiers]

    for i in range(prediction_interval):
        df_labels[f"T+{i+1}"] = df_labels["datetime"] + pd.DateOffset(days=i+1)
    
    df_labels_view = df[identifiers + ["rlf"]]
    for i in range(prediction_interval):
        target_day_column_name = f"T+{i+1}"

        df_labels = df_labels.merge(df_labels_view, 
                  how = "left", 
                  left_on = ("site_id", "mlid", target_day_column_name),
                  right_on = identifiers,
                  suffixes = ("", "_y")
        )
        df_labels.rename(columns={"rlf": f"{target_day_column_name}_rlf"}, inplace=True)

    df_labels.drop(columns=["datetime_y"], inplace=True)

    df_labels["1-day-predict"] = df_labels["T+1_rlf"]

    df_labels["5-day-predict"] = df_labels[[f"T+{i+1}_rlf" for i in range(prediction_interval)]].any(axis=1)

    df_labels = df_labels[["datetime", "site_id", "mlid", "1-day-predict", "5-day-predict"]]

    rl_kpis_with_labels = df.merge(df_labels, 
                                    how="left", 
                                    on=["datetime", "site_id", "mlid"])
    
    categorical_features = ["card_type", "freq_band", "type", "tip", "adaptive_modulation", "freq_band", "modulation"]

    numerical_features = ["severaly_error_second", "error_second", "unavail_second", "avail_time", "bbe", "rxlevmax", "capacity"]

    labels = ["rlf", "1-day-predict", "5-day-predict"]

    rl_kpis_with_labels = rl_kpis_with_labels.dropna()

    rl_kpis_with_labels[labels] = rl_kpis_with_labels[labels].astype(int)

    one_hot_encoder = OneHotEncoder(handle_unknown='error')
    one_hot_encoder.fit(rl_kpis_with_labels[categorical_features])
    rl_kpis_with_labels[one_hot_encoder.get_feature_names_out()] = one_hot_encoder.transform(rl_kpis_with_labels[categorical_features]).toarray()

    rl_kpis_with_labels = rl_kpis_with_labels.drop(columns=categorical_features)
    categorical_features = list(one_hot_encoder.get_feature_names_out())

    numerical_dataset = rl_kpis_with_labels.loc[:, identifiers + numerical_features + labels]

    categorical_dataset = rl_kpis_with_labels.loc[:, identifiers + categorical_features + labels]

    features = numerical_features.copy()

    for feature in numerical_features:
        historical_num_dataset = numerical_dataset.loc[:, identifiers]
        for i in range(-1,  -look_back_interval, -1):
            historical_num_dataset[f"T{i}"] = historical_num_dataset["datetime"] + pd.DateOffset(days=i)

        numeric_view = numerical_dataset[identifiers + [feature]]
        for i in range(-1,  -look_back_interval, -1):
            target_day_column_name = f"T{i}"

            historical_num_dataset = historical_num_dataset.merge(numeric_view, 
                    how = "left", 
                    left_on = ("site_id", "mlid", target_day_column_name),
                    right_on = identifiers,
                    suffixes = ("", "_y")
            )
            historical_num_dataset.rename(columns={ feature: f"{target_day_column_name}_{feature}"}, inplace=True)
            features.append(f"{target_day_column_name}_{feature}")

        historical_num_dataset.drop(columns=["datetime_y"], inplace=True)

        historical_num_dataset = historical_num_dataset.drop(columns=[f"T{i}" for i in range(-1,  -look_back_interval, -1)])

        numerical_dataset = numerical_dataset.merge(historical_num_dataset, 
                    how="left", 
                    on=["datetime", "site_id", "mlid"])
        
    numerical_dataset = numerical_dataset.dropna()
    numerical_dataset = numerical_dataset[identifiers + features + labels]

    numerical_dataset.to_csv(output/"preprocessed_numerical_features.csv", index=None)
    categorical_dataset.to_csv(output/"preprocessed_categorical_features.csv", index=None)

    print(numerical_dataset.columns)
    print(numerical_dataset.head())

preprocessing("train/rl-kpis.tsv", "train/distances.tsv", "met-forecast.tsv", "met-real.tsv", ".")