# Stock Price Forecasting — AMZN & MSFT

## Overview
This was a group project for DAMO-611 (Data Analytics Case Study 3) at the University of Niagara Falls Canada. The objective was to predict the 30-day forward stock performance of Amazon (AMZN) and Microsoft (MSFT) using five years of historical data, and to evaluate which stock presented stronger near-term returns from a corporate cash management perspective. The business framing was deliberate — this was not an academic exercise in model comparison but a simulation of what an asset management team would actually want: an accurate, uncertainty-minimized forecasting mechanism to support short-term investment decisions.

Six models were built, tuned, and evaluated on the same 30-day holdout test set. XGBoost won.

## What I Worked On
I led the Exploratory Data Analysis pipeline, which in a time-series forecasting project is not a preliminary step — it is the foundation that every model depends on. I pulled five years of AMZN and MSFT closing price and volume data programmatically using yfinance, handled missing values using backward fill, and built the full feature engineering layer: daily return percentages, 30-day and 90-day moving averages, and drawdown metrics tracking peak-to-trough severity over time. Getting the moving averages and drawdown calculations right required thinking carefully about lookahead bias — making sure no future data was leaking into the features used to train the models.

I built the core diagnostic visualizations used throughout the project: normalized price overlays showing both stocks indexed to the same starting point, return distribution histograms with fitted normal curves, drawdown charts, Pearson correlation analysis on both raw closing prices and daily returns, and annual calendar-year return bar charts covering 2022 through 2025. These were not just exploratory — several of them directly informed which models made sense to try and what the training window should look like.

I also conducted code review across the full notebook to make sure the modeling pipelines were implemented consistently and that the evaluation metrics were being computed correctly. On the documentation side, I wrote the methodology and results sections covering the modeling phase — translating what the error metrics actually meant in the context of two very differently-priced stocks (which is why we computed relative MAE, MSE, and RMSE divided by mean price, not just the raw values).

## The Six Models
All six models were trained on the same data and evaluated on the final 30 business days as an unseen test block.

**Exponential Smoothing (Holt-Winters)** — baseline model with additive trend. Useful as a benchmark but not built to capture the kind of non-linear patterns in equity prices.

**Modified Exponential Smoothing** — extended version with a damped trend, additive seasonality, and a 365-day seasonal period. Better than the baseline but still fundamentally a smoothing approach.

**SARIMA** — autoregressive integrated moving average with a seasonal component. Captures autocorrelation structure in the price series but assumes linearity that equity data does not always respect.

**SARIMAX** — SARIMA with trading volume added as an exogenous variable. The hypothesis was that volume spikes precede price movements. It helped at the margins but did not dramatically change the error picture.

**XGBoost Regressor** — the clear winner. Trained on 30 lag features (the previous 30 closing prices), predicting one day forward recursively. Gradient boosting captured non-linear relationships in the price history that the statistical models could not. Achieved approximately 2% MAPE on both stocks.

**Prophet (Meta)** — useful for visualizing long-range trend boundaries and changepoints but less suited to the short 30-day evaluation window we were working with.

## Key Findings
- XGBoost was the best-performing model across all four error metrics (MAE, MAPE, MSE, RMSE) for both stocks
- MAPE of approximately 2% on both AMZN and MSFT — comparable accuracy despite the significant difference in raw share price between the two
- Cohen's d and difference-of-means tests indicated MSFT as the marginally safer analytical pick, though the statistical divergence between the two forecasted outcomes was not large enough to be decisive
- The relative error metrics (errors divided by mean price) were essential for making the comparison meaningful — raw RMSE numbers are not comparable between a $180 stock and a $200+ stock

## Limitations
The models were trained and evaluated on historical price data, which captures patterns but not events. Any major earnings surprise, macroeconomic shock, or sector-level news during the 30-day forecast window would not be reflected in the model's predictions. The XGBoost model in particular is recursive — errors in day 1's prediction feed into day 2's input, which means errors compound over the forecast horizon. For very short windows (5–10 days) this is not a major issue, but at 30 days the compounding effect is noticeable. A production-grade version of this would likely cap the recursive window and retrain more frequently.

## Data Sources
- Yahoo Finance via yfinance Python library (March 2021 – March 2026)

## Tech Stack
Python, XGBoost, StatsModels (SARIMA/SARIMAX), Prophet, Holt-Winters (statsmodels), yfinance, Pandas, NumPy, Matplotlib, Scikit-learn, Tableau, Jupyter

## How to Run

1. Clone the repository
2. Run `pip install jupyter pandas numpy matplotlib scikit-learn xgboost statsmodels prophet yfinance`
3. Open `notebooks/stock_forecasting_amzn_msft.ipynb` in Jupyter
4. Run all cells top to bottom — the notebook pulls live data from Yahoo Finance so no data files are needed
5. To view the Tableau visualizations, open `tableau/stock_analysis.twbx` in Tableau Desktop or Tableau Public (free)

Note: Prophet requires a clean installation environment. If you run into installation issues on Windows, try installing it in a conda environment with `conda install -c conda-forge prophet`.

## What I Would Do Next
- Extend the model comparison to include an LSTM or Transformer-based architecture to see whether deep learning approaches outperform XGBoost on longer forecast horizons.
- Add a proper walk-forward validation setup rather than a single train/test split — this would give a much more reliable estimate of real-world model performance across different market regimes.
- Incorporate sentiment data from financial news or earnings call transcripts as additional exogenous features, which might help the model respond to information that pure price history cannot capture.
