# Price Forecasting Exercise

This task involves building a Python program to forecast electricity prices using a dataset containing 15-minute interval electricity prices and ambient temperature for Barstow, California. The column rt_price is known as Locational Marginal Price, which is the price of electricity at a specific location in the power grid at a particular point in time. The column dah_price is the day-ahead electricity price and refers to the price at which electricity is bought and sold for delivery the following day. The program should implement a forecast model that considers temperature and predicts the LMP price 24 steps into the future (equivalent to 6 hours). Additionally, it should provide accuracy metrics such as Root Mean Squared Error (RMSE), Percentage Error (PE), Mean Absolute Percentage Error (MAPE), etc. Furthermore, the program should compare the accuracy between on-peak hours (8 am – midnight) and off-peak hours (midnight-8 am).

Please include a summary of your analysis including an explanation of your methodology and key findings. Feel free to include visualizations if relevant. The team should be able to run your code and reproduce results out of the box. If additional setup (third party libraries etc.) is required, please include it in the summary.

## Summary

- Started at 11 PM Tuesday May 7
- Convert date column to datetime index
- Optionally generate Auto-EDA interactive html document
- Initialize ARIMAX (Auto-Regressive Moving Average with eXogenous inputs) model using `temp_BAR`, `hour`, `month`, `season`
- Deseasonalize temp_BAR using a periodicity of 1 month
- Create `sklearn.model_selection.TimeSeriesSplit` Cross Validator, since no validation set was given to me. 
- In each split I do the following:
  0. Create my baseline ARIMAX model. Based on ADF test as well as ACF and PACF graphs, I chose ordering values p=5, d=0, q=1.
  1. Create a `future_exog` dataframe that represents the extruded exogenous variables I need to forecast the next 24 hours of `rt_energy` with my ARIMAX model
  2. Create ARMA model for `temp_BAR` in this split using hand-selected lag values informed by running an Augmented Dickey-Fuller test for stationarity as well as analyzing ACF and PACF graphs to pick the best autoregressive lag values and moving average lag values.
  3. Algorithmically extrude the datetimes by the forecast horizon, as well as derive selected exogenous variables from the datetime index
  4. Predict next 24 hours worth of `temp_BAR` values as per above
  5. Take our original fitted ARIMAX and predict the next 24 hours worth of `rt_energy`
  6. Calculate metrics and visualize model performance 

## Installation

1. Clone the repository
2. run `python -m venv .venv`
3. run `pip install -r requirements.txt`
4. add dataset to `/data`
5. 

