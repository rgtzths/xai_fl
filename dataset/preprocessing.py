import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pathlib
import gc

def finding_closest_stations(rl_sites_file, distances_file, met_stations_file, k=1):
    rl_sites_df = pd.read_csv(rl_sites_file, sep="\t", index_col=0)
    distances_df = pd.read_csv(distances_file, sep="\t", index_col=0)
    met_stations_df = pd.read_csv(met_stations_file, sep="\t", index_col=0)

    rl_stations = rl_sites_df["site_id"].unique()
    met_stations = met_stations_df["station_no"].unique()

    distances_df = distances_df.loc[met_stations, rl_stations]

    closest_stations = dict(distances_df.idxmin())

    return closest_stations



def preprocessing(kpi_file, distances_file, met_forecast_file, rl_sites_file, met_stations_file, output_path, prediction_interval=5, look_back_interval=10):
    output = pathlib.Path(output_path)
    output.mkdir(parents=True, exist_ok=True)

    closest_stations = finding_closest_stations(rl_sites_file, distances_file, met_stations_file)

    identifiers = ["site_id", "mlid", "datetime"]
    df = pd.read_csv(kpi_file, sep="\t", index_col=0)

    if "datetime" in df:
        df["datetime"] = pd.to_datetime(df["datetime"])

    met_forecast_df = pd.read_csv(met_forecast_file, sep="\t", index_col=0)
    if "datetime" in met_forecast_df:
        met_forecast_df["datetime"] = pd.to_datetime(met_forecast_df["datetime"])
    
    met_forecast_df = met_forecast_df[ met_forecast_df["report_time"] == "morning"]
    met_forecast_df.drop(columns=["report_time"], inplace=True)

    df["station_no"] = [closest_stations[site_id] for  site_id in df["site_id"]]

    df = df.merge(met_forecast_df, how="left", on=["datetime", "station_no"])
    
    del met_forecast_df
    del closest_stations

    df.drop(columns=["station_no", "mw_connection_no"], inplace=True)

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
    del df_labels

    static_features = ["card_type", "freq_band", "type", "tip", "adaptive_modulation", "freq_band", "modulation"]
    labels = ["rlf", "1-day-predict", "5-day-predict"]
    time_sentitive_features = [feature for feature in df.columns if feature not in static_features and feature not in labels and feature not in identifiers]

    #Check for nans in static_features
    one_hot_encoder = OneHotEncoder(handle_unknown='error')
    one_hot_encoder.fit(rl_kpis_with_labels[static_features])
    rl_kpis_with_labels[one_hot_encoder.get_feature_names_out()] = one_hot_encoder.transform(rl_kpis_with_labels[static_features]).toarray()

    rl_kpis_with_labels.drop(columns=static_features, inplace=True)
    static_features = list(one_hot_encoder.get_feature_names_out())

    time_sentitive_dataset = rl_kpis_with_labels.loc[:, identifiers + time_sentitive_features + labels]

    static_dataset = rl_kpis_with_labels.loc[:, identifiers + static_features + labels]
    
    del rl_kpis_with_labels

    for feature in time_sentitive_features:
        historical_sen_dataset = time_sentitive_dataset.loc[:, identifiers]
        for i in range(-1,  -look_back_interval, -1):
            historical_sen_dataset[f"T{i}"] = historical_sen_dataset["datetime"] + pd.DateOffset(days=i)

        feature_view = time_sentitive_dataset[identifiers + [feature]]
        for i in range(-1,  -look_back_interval, -1):
            target_day_column_name = f"T{i}"

            historical_sen_dataset = historical_sen_dataset.merge(feature_view, 
                    how = "left", 
                    left_on = ("site_id", "mlid", target_day_column_name),
                    right_on = identifiers,
                    suffixes = ("", "_y")
            )
            historical_sen_dataset.rename(columns={ feature: f"{target_day_column_name}_{feature}"}, inplace=True)
        
        del feature_view

        historical_sen_dataset.drop(columns=["datetime_y"], inplace=True)

        historical_sen_dataset.drop(columns=[f"T{i}" for i in range(-1,  -look_back_interval, -1)], inplace=True)

        time_sentitive_dataset = time_sentitive_dataset.merge(historical_sen_dataset, 
                    how="left", 
                    on=["datetime", "site_id", "mlid"])
        
        del historical_sen_dataset
        gc.collect()

    features = time_sentitive_features.copy()
    for i in range(-1,  -look_back_interval, -1):
        for feature in time_sentitive_features:
            features.append(f"T{i}_{feature}")

    time_sentitive_dataset = time_sentitive_dataset.dropna()
    time_sentitive_dataset[labels] = time_sentitive_dataset[labels].astype(int)

    weather_features = []

    for j in range(0, -look_back_interval, -1):
        for i in range(1, 6, 1):
            if j < 0:
                weather_features.append(f"T{j}_weather_day{i}")
            else:
                weather_features.append(f"weather_day{i}")


    one_hot_encoder.fit(time_sentitive_dataset[weather_features])
    time_sentitive_dataset[one_hot_encoder.get_feature_names_out()] = one_hot_encoder.transform(time_sentitive_dataset[weather_features]).toarray()

    time_sentitive_dataset.drop(columns=weather_features, inplace=True)
    time_sentitive_features = [feature for feature in time_sentitive_dataset.columns if feature not in labels and feature not in identifiers]

    time_sentitive_dataset = time_sentitive_dataset[identifiers + time_sentitive_features + labels]

    time_sentitive_dataset.to_csv(output/"preprocessed_time_sentitive_features.csv", index=None)
    static_dataset.to_csv(output/"preprocessed_static_features.csv", index=None)

    print(time_sentitive_dataset.columns)
    print(time_sentitive_dataset.head())

preprocessing("train/rl-kpis.tsv", "train/distances.tsv", "train/met-forecast.tsv", "train/rl-sites.tsv",  "train/met-stations.tsv",".")

