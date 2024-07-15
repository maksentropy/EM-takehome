# EM-takehome

My idea was to design a cascade of a ML system, which is maintainable and can be extended in future if needed. There are two main parts in the system:

Script `model.py` contains a class wrapper for a ML model. Right now it relies on sklearn objects and the functions resemble functionality of sklearn objects, which might seem redundant. The idea was to write this class to hide all details of using other ML frameworks (or even own implementations) if it's needed at some point in the future.

Right now the class cointains 3 methods: 
- `train_model` to train or retrain the model. 
- `predict` to make preductions for the given data.
- `score` for monitoring accuracy of a model. Since it's a regression model two metrics were chosen: mean absolute error (MAE) and mean absolue persentage error (MAPE).


`data_processing.py` script containts a class wrapper for a data class. `DataProcesser` extends functionality of `pandas` library with two methods:
- `retreive_data` for reading data. Right now it reads data from a CSV file but it can be extended to get data from a database or API endpoint.
- `fill_na` is for dealing with missing values. It implements filling missed numerical data with median.

`predictions.py` scipt trains the model and retunrs predictions. 

For machine learning model part I assumed that only variables with prefix `latest_forecasted_*` are available when forecast for carbon intensity is made for future. This is how the distribution of missing values looks like.

```
Column                                         Non-Null Count  Dtype  
---  ------                                         --------------  -----  
 0   latest_forecasted_dewpoint_avg                 25235 non-null  float64
 1   latest_forecasted_precipitation_avg            24901 non-null  float64
 2   latest_forecasted_solar_avg                    29566 non-null  float64
 3   latest_forecasted_temperature_avg              25174 non-null  float64
 4   latest_forecasted_wind_x_avg                   29579 non-null  float64
 5   latest_forecasted_wind_y_avg                   29579 non-null  float64
 6   latest_forecasted_price_avg                    41734 non-null  float64
 7   latest_forecasted_production_avg               41205 non-null  float64
 8   latest_forecasted_consumption_avg              41733 non-null  float64
 9   latest_forecasted_power_net_import_DE_avg      41445 non-null  float64
 10  latest_forecasted_power_net_import_DK-DK1_avg  41325 non-null  float64
 11  latest_forecasted_power_net_import_SE_avg      0 non-null      float64
 12  latest_forecasted_power_net_import_SE-SE4_avg  41770 non-null  float64
 13  latest_forecasted_production_solar_avg         41301 non-null  float64
 14  latest_forecasted_production_wind_avg          41301 non-null  float64
```

The column `latest_forecasted_power_net_import_SE_avg` can be since there are no values at all. Since there are observations with approximately half of missing values, filling them with median value might be tricky. On the other hand, there are approximately 25000 observations if we drop missing values. 

To compare models, the dataset is split into training and test sets with proportions 80/20 % of the total dataset size. Baseline model is predicting mean of training set. Metric - mean absolute error. 

Here is results of training several models:
| Model | MAE |
| --- | --- |
| Baseline (mean) | 108.0245 |
| Ridge regression | 46.9479 |
| Lasso regression | 48.4254 |
| Gradient boosting | 49.3276 |

However, sklearn's implementation of gradient boosting has property of showing feature importance, which puts most attention to `latest_forecasted_production_avg`, `latest_forecasted_power_net_import_DE_avg`, `latest_forecasted_power_net_import_SE-SE4_avg`,`latest_forecasted_production_wind_avg`. Selecting these features and training the gradient boosting only with these features yields the following results. 

| Model | MAE |
| --- | --- |
| Baseline (mean) | 108.0245 |
| Ridge regression | 46.9479 |
| Lasso regression | 48.4254 |
| Gradient boosting | 49.3276 |
| Gradient boosting (selected features) | 41.16141 |

Gradient boosting with selected features is used in the ML pipeline in this repo. 