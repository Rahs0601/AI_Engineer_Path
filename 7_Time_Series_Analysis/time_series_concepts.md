# Time Series Analysis Concepts

Time series analysis involves methods for analyzing time series data in order to extract meaningful statistics and other characteristics of the data. It is used to forecast future values based on previously observed values.

## Key Concepts:

### 1. What is a Time Series?

*   A sequence of data points indexed (or listed or graphed) in time order.
*   Most commonly, a time series is a sequence taken at successive equally spaced points in time.
*   Examples: Stock prices, daily temperature, monthly sales, sensor readings.

### 2. Components of a Time Series

*   **Trend**: A long-term increase or decrease in the data.
*   **Seasonality**: A pattern that repeats over a fixed period (e.g., daily, weekly, monthly, yearly).
*   **Cyclical**: Patterns that are not of a fixed period, often associated with business cycles.
*   **Irregular/Residual**: Random variations or noise in the data after accounting for trend, seasonality, and cyclical components.

### 3. Stationarity

*   A time series is stationary if its statistical properties (mean, variance, autocorrelation) do not change over time.
*   Many time series models assume stationarity.
*   **Tests for Stationarity**: Augmented Dickey-Fuller (ADF) test, Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test.
*   **Making a Series Stationary**: Differencing (subtracting the previous value from the current value), transformations (e.g., log transform).

### 4. Autocorrelation and Partial Autocorrelation

*   **Autocorrelation Function (ACF)**: Measures the correlation between a time series and a lagged version of itself. Helps identify seasonality and trend.
*   **Partial Autocorrelation Function (PACF)**: Measures the correlation between a time series and a lagged version of itself, after controlling for the effects of intermediate lags. Helps identify the order of AR (AutoRegressive) terms.

## 5. Traditional Time Series Models

*   **Moving Average (MA)**: Forecasts based on the average of past errors.
*   **AutoRegressive (AR)**: Forecasts based on a linear combination of past values of the variable.
*   **AutoRegressive Moving Average (ARMA)**: Combines AR and MA models. Requires stationary data.
*   **AutoRegressive Integrated Moving Average (ARIMA)**: An extension of ARMA that handles non-stationary data by differencing it (the "I" stands for Integrated).
    *   **p**: Order of the AR part.
    *   **d**: Order of differencing.
    *   **q**: Order of the MA part.
*   **Seasonal ARIMA (SARIMA)**: An extension of ARIMA that supports the direct modeling of the seasonal component of the series.
*   **Exponential Smoothing (ETS)**: Models that assign exponentially decreasing weights over time. Includes Simple Exponential Smoothing, Holt's Linear Trend, and Holt-Winters (for seasonality).
*   **Prophet**: A forecasting procedure developed by Facebook that is optimized for business forecasts. It handles seasonality, holidays, and missing data well.

## 6. Deep Learning for Time Series

Deep learning models are increasingly used for time series forecasting, especially for complex patterns and large datasets.

*   **Recurrent Neural Networks (RNNs)**: Particularly LSTMs and GRUs, are well-suited for sequential data due to their ability to capture temporal dependencies.
*   **Convolutional Neural Networks (CNNs)**: Can be used for feature extraction from time series data, especially 1D CNNs.
*   **Transformers**: Originally for NLP, Transformers are now being applied to time series for their ability to capture long-range dependencies and parallelize computations.
*   **Hybrid Models**: Combining traditional statistical methods with deep learning (e.g., ARIMA + LSTM).

## 7. Evaluation Metrics for Time Series Forecasting

*   **Mean Absolute Error (MAE)**
*   **Mean Squared Error (MSE)**
*   **Root Mean Squared Error (RMSE)**
*   **Mean Absolute Percentage Error (MAPE)**

## Resources:

*   **"Forecasting: Principles and Practice" by Rob J Hyndman and George Athanasopoulos**
*   **"Time Series Analysis and Its Applications: With R Examples" by Robert H. Shumway and David S. Stoffer**
*   **Facebook Prophet Documentation**
*   **Online courses on Time Series Analysis**
