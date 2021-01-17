# IntelliTrader
An AI Leveraged Financial Trading System FX-USD/Euro

# Key words:
Trading – USD/Eur – FX prediction – Traded data (Ticker data) - Sentiment analysis – Feature Extraction – Statistical forecasting - Neural networks – Prediction accuracy

# Context: 
Financial markets asset pricing has been a subject of immense research for being able to predict has immense financial rewards. Financial Assets include Foreign exchange (FX), bonds, equities etc.

Financial assets prices are influenced by various influencing factors covering economic, political, social, and regulatory dynamic. Financial assets trading with a higher temporal prediction accuracy leads to financially superior trade outcomes.
# Aims: 
We aim to predict FX for next day trading session – i.e. Predict today (T0), tomorrow’s /next trading day/rolling 24 hrs FX rate (T0+1). We try to predict USD/Euro fiat currencies as they constitute the most traded and dynamic currency pair.
The objective is to create a prediction system that helps market participant – i.e. a FX broker. Focus is on prediction accuracy Vs specific broker platforms. Specific broker platforms will be identified later, based on the quality of their APIs, features offered and trade execution.
The goal of the project is to achieve high enough next day prediction accuracy for a successful trading position over a period of time.

# Methods: 
We collected historical FX transaction data over several preceding years. We defined and engineered features from the raw datasets that heuristically synthesize and summarize information. We set the transaction data (/last week /last fortnight /last month /last quarter) sets. We researched a variety of statistical and Deep Neural Networks based models. We didn’t use text data as part of model building for commercial reasons. The pipelines were duly built.
We trained multiple deep learning models and received varying prediction accuracy.
Overall, we built the complete pipelines for data ingestion, data cleaning, feature engineering, ongoing training, so that a commercial trader can be served daily feeds for next day’s trading session.

# Conclusions:
Model has successfully learnt predicting next day FX. While we will back-tested and took care to avoid overfitting, the model performance underperforms our expectations for being used as a decision support system for a trading desk.
Our Prediction model lacks incorporation of market news and market sentiments. Surveying various literatures as well as applying our own experience, we see that incorporation of same should improve the prediction accuracy substantially.
We aim to improve the model further by exploiting two areas:
a) Embed trading community sentiment and context News besides trading data.
b) Experiment further novel information extraction networks
