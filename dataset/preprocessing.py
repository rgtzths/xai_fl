import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pathlib

def finding_closest_stations(rl_sites_file, distances_file, met_stations_file, k=1):
    rl_sites_df = pd.read_csv(rl_sites_file, sep="\t", index_col=0)
    distances_df = pd.read_csv(distances_file, sep="\t", index_col=0)
    met_stations_df = pd.read_csv(met_stations_file, sep="\t", index_col=0)

    rl_stations = rl_sites_df["site_id"].unique()
    met_stations = met_stations_df["station_no"].unique()

    distances_df = distances_df.loc[met_stations, rl_stations]

    closest_stations = dict(distances_df.idxmin())

    return closest_stations


input_path = "train"
output_path ="./preprocessed_train_met/"

prediction_interval=5
look_back_interval=10
all_weather = True

output = pathlib.Path(output_path)
output.mkdir(parents=True, exist_ok=True)
input = pathlib.Path(input_path)

kpi_file= input/"rl-kpis.tsv"
distances_file = input/"distances.tsv"
met_forecast_file = input/"met-forecast.tsv"
rl_sites_file = input/"rl-sites.tsv"
met_stations_file = input/"met-stations.tsv"


one_hot_encoder = OneHotEncoder(handle_unknown='error')

#Get the closest met station to each radio link one.
closest_stations = finding_closest_stations(rl_sites_file, distances_file, met_stations_file)

#List of columns that uniquelly identify an entry in the kpi_df
identifiers = ["site_id", "mlid", "datetime"]

kpi_df = pd.read_csv(kpi_file, sep="\t", index_col=0)
met_forecast_df = pd.read_csv(met_forecast_file, sep="\t", index_col=0)

#Transform the datetime column to the correct format
kpi_df["datetime"] = pd.to_datetime(kpi_df["datetime"])

met_forecast_df["datetime"] = pd.to_datetime(met_forecast_df["datetime"])

# Filtering the reports to only include the morning report, and removing that column afterwards. (usually there are morning and afternoon reports)
met_forecast_df = met_forecast_df[ met_forecast_df["report_time"] == "morning"]
met_forecast_df.drop(columns=["report_time"], inplace=True)

if not all_weather:
    columns = [column for column in met_forecast_df.columns if "day5" not in column and column not in ["station_no", "datetime"]]
    met_forecast_df.drop(columns=columns, inplace=True)

#Adding closest station to each entry according to the id of the radio link
kpi_df["station_no"] = [closest_stations[site_id] for  site_id in kpi_df["site_id"]]

# Merging kpis with forecast data
kpi_df = kpi_df.merge(met_forecast_df, how="left", on=["datetime", "station_no"])

#Remove unnecessary columns
kpi_df.drop(columns=["station_no", "mw_connection_no"], inplace=True)

#kpi_df.drop(columns=["mw_connection_no"], inplace=True)

if "scalibility_score" in kpi_df.columns:
    kpi_df.drop(columns=["scalibility_score"], inplace=True)

if all_weather:
    weather_features = [f"weather_day{i}" for i in range(1,6)]
else:
    weather_features = ["weather_day5"]

classes = np.array(['light rain', 'rain', 'heavy rain showers', 'hot day', 'scattered clouds',
 'light rain showers', 'nan', 'overcast clouds',
 'thunderstorm with heavy rain', 'foggy', 'few clouds', 'heavy rain', 'windy',
 'snow', 'misty', 'sleet', 'light intensity shower rain', 'clear sky',
 'light snow', 'heavy thunderstorm with rain showers'])

classes = classes.reshape(-1, 1)

one_hot_encoder.fit(classes)
for weather_feature in weather_features:
    columns = one_hot_encoder.get_feature_names_out([weather_feature])
    kpi_df = pd.concat([kpi_df,
                pd.DataFrame(one_hot_encoder.transform(kpi_df[weather_feature].to_numpy(dtype=str).reshape(-1,1)).toarray(),
                            columns=columns)
                ],
                axis=1)

kpi_df.drop(columns=weather_features, inplace=True)


df_labels = kpi_df.loc[:, identifiers]

for i in range(prediction_interval):
    df_labels[f"T+{i+1}"] = df_labels["datetime"] + pd.DateOffset(days=i+1)

df_labels_view = kpi_df[identifiers + ["rlf"]]
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

kpi_df = kpi_df.merge(df_labels, 
                                how="left", 
                                on=["datetime", "site_id", "mlid"])


static_features = ["card_type", "freq_band", "type", "tip", "adaptive_modulation", "freq_band", "modulation"]
labels = ["rlf", "1-day-predict", "5-day-predict"]
time_sentitive_features = [feature for feature in kpi_df.columns if feature not in static_features and feature not in labels and feature not in identifiers]


time_sentitive_dataset = kpi_df.loc[:, identifiers + time_sentitive_features + labels]

static_dataset = kpi_df.loc[:, identifiers + static_features + labels]
static_dataset = static_dataset.dropna()

one_hot_encoder.fit(static_dataset[static_features])

static_dataset = pd.concat([static_dataset,
                            pd.DataFrame(one_hot_encoder.transform(static_dataset[static_features]).toarray(),
                                                         columns=one_hot_encoder.get_feature_names_out())
                            ],
                        axis=1
                        )

static_dataset.drop(columns=static_features, inplace=True)

static_dataset.to_csv(output/"preprocessed_static_features.csv", index=None)

normalizable_features = [feature for feature in time_sentitive_features if "weather_day" not in feature]
max_values = time_sentitive_dataset[normalizable_features].max()
min_values = time_sentitive_dataset[normalizable_features].min()

time_sentitive_dataset[normalizable_features] = (time_sentitive_dataset[normalizable_features] - min_values) / (max_values - min_values)

base_features = [feature for feature in time_sentitive_dataset.columns if feature not in labels and feature not in identifiers and "T-" not in feature and "nan" not in feature]

ordered_features = []
for i in range(-look_back_interval +1, 0, 1):
    for feature in base_features:
        ordered_features.append(f"T{i}_{feature}")
ordered_features += base_features

rl_mlid_combos =  (time_sentitive_dataset["site_id"] + "%" + time_sentitive_dataset["mlid"]).unique()


weather_nan_features = []

if all_weather:
    for i in range(1,6):
        for j in range(0,  -look_back_interval, -1):
            if j == 0:
                weather_features.append(f"weather_day{i}_nan")
            else:
                weather_features.append(f"T{j}_weather_day{i}_nan")
else:
    for j in range(0,  -look_back_interval, -1):
        if j == 0:
            weather_features.append(f"weather_day5_nan")
        else:
            weather_features.append(f"T{j}_weather_day5_nan")

rl_sites_df = pd.read_csv(rl_sites_file, sep="\t", index_col=0)
rl_sites_df["clutter_class"] = [ x.replace(" ", "_") for x in rl_sites_df["clutter_class"]]
site_clutter = dict(rl_sites_df[ ["site_id", "clutter_class"]].values)

for rl_mlid in rl_mlid_combos:
    try:
        site_id, mlid = rl_mlid.split("%")

        rl_mlid_df = time_sentitive_dataset.loc[(time_sentitive_dataset["site_id"] == site_id) & (time_sentitive_dataset["mlid"] == mlid)]

        for feature in time_sentitive_features:
            historical_sen_dataset = rl_mlid_df.loc[:, identifiers]
            for i in range(-1,  -look_back_interval, -1):
                historical_sen_dataset[f"T{i}"] = historical_sen_dataset["datetime"] + pd.DateOffset(days=i)

            feature_view = rl_mlid_df[identifiers + [feature]]
            for i in range(-1,  -look_back_interval, -1):
                target_day_column_name = f"T{i}"

                historical_sen_dataset = historical_sen_dataset.merge(feature_view, 
                        how = "left", 
                        left_on = ("site_id", "mlid", target_day_column_name),
                        right_on = identifiers,
                        suffixes = ("", "_y")
                )
                historical_sen_dataset.rename(columns={ feature: f"{target_day_column_name}_{feature}"}, inplace=True)

            historical_sen_dataset.drop(columns=["datetime_y"], inplace=True)

            historical_sen_dataset.drop(columns=[f"T{i}" for i in range(-1,  -look_back_interval, -1)], inplace=True)

            rl_mlid_df = rl_mlid_df.merge(historical_sen_dataset, 
                        how="left", 
                        on=["datetime", "site_id", "mlid"])


        #Remove nan weather columns and positive entries
        for weather_feature in weather_nan_features:
            rl_mlid_df = rl_mlid_df.loc[rl_mlid_df[weather_feature] != 1]
            rl_mlid_df.drop(columns=[weather_feature], inplace=True)

        rl_mlid_df = rl_mlid_df.dropna()
        rl_mlid_df[labels] = rl_mlid_df[labels].astype(int)

        rl_mlid_df = rl_mlid_df[identifiers + ordered_features + labels]

        clutter_folder = output / site_clutter[site_id]
        site_folder = clutter_folder / site_id
        site_folder.mkdir(parents=True, exist_ok=True)

        rl_mlid_df.to_csv(site_folder/f"{mlid}_time_sentitive_features.csv", index=None)
    except:
        print(rl_mlid)
