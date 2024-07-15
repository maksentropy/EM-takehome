# EM-takehome

My idea was to design a cascade of a ML system, which is maintainable and can be extended in future if needed. There are two main parts in the system:

Script `model.py` contains a class wrapper for a ML model. Right now it relies on sklearn objects and the functions resemble functionality of sklearn objects, which might seem redundant. The idea was to write this class to hide all details of using other ML frameworks (or even own implementations) if it's needed at some point in the future.

Right now the class cointains 3 methods: 
    * `train_model` to train or retrain the model. 
    * `predict` to make preductions for the given data.
    * `score` for monitoring accuracy of a model. Since it's a regression model two metrics were chosen: mean absolute error (MAE) and mean absolue persentage error (MAPE).

`data_processing.py` script containts a class wrapper for a data class. `DataProcesser` extends functionality of `pandas` library with two methods:
    * `retreive_data` for reading data. Right now it reads data from a CSV file but it can be extended to get data from a database or API endpoint.
    * `fill_na` is for dealing with missing values. It implements filling missed numerical data with median.

`predictions.py` scipt trains the model and retunrs predictions. 

For machine learning model part I assumed that only variables with prefix `latest_forecasted_*` are available when forecast for carbon intensity is made for future. 