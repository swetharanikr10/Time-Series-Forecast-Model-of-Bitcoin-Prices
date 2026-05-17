

#Time Series Forecast Model of Bitcoin Prices

#Predictive Analytics for Financial Decision-Making


# Packages
library(tidyverse)
library(forecast)
library(tseries)
library(lubridate)
library(zoo)
library(urca)
library(readr)


# 1. Load data

btc <- read_csv("Bitcoin_Price_Dataset_2014_2023.csv")

str(btc)
names(btc)


# 2. Clean data

btc <- btc %>%
  rename(
    Date = 1,
    Open = 2,
    High = 3,
    Low = 4,
    Close = 5,
    Volume = 6
  ) %>%
  mutate(
    Date = as.Date(Date),
    Open = as.numeric(Open),
    High = as.numeric(High),
    Low = as.numeric(Low),
    Close = as.numeric(Close),
    Volume = as.numeric(Volume)
  ) %>%
  arrange(Date) %>%
  drop_na()


# 3. Descriptive statistics

summary(btc[, c("Close", "Volume")])
sd(btc$Close, na.rm = TRUE)
sd(btc$Volume, na.rm = TRUE)

descriptive_stats <- btc %>%
  summarise(
    Close_Mean = mean(Close, na.rm = TRUE),
    Close_SD = sd(Close, na.rm = TRUE),
    Close_Min = min(Close, na.rm = TRUE),
    Close_Max = max(Close, na.rm = TRUE),
    Volume_Mean = mean(Volume, na.rm = TRUE),
    Volume_SD = sd(Volume, na.rm = TRUE),
    Volume_Min = min(Volume, na.rm = TRUE),
    Volume_Max = max(Volume, na.rm = TRUE)
  )

print(descriptive_stats)


# 3A. Boxplots

btc %>%
  select(Open, High, Low, Close, Volume) %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value") %>%
  ggplot(aes(x = Variable, y = Value, fill = Variable)) +
  geom_boxplot() +
  labs(title = "Boxplots of Bitcoin Variables",
       x = "Variables", y = "Values") +
  theme_minimal() +
  theme(legend.position = "none")


# 4. Time series object

btc_ts <- ts(btc$Close, frequency = 7)


# 5. Plot time series

autoplot(btc_ts) +
  labs(title = "Bitcoin Daily Closing Prices",
       x = "Time", y = "Close Price")


# 6. Correlation and lag plots

Acf(btc_ts, main = "ACF of Bitcoin Closing Prices")
Pacf(btc_ts, main = "PACF of Bitcoin Closing Prices")

btc <- btc %>%
  mutate(
    Close_Lag1 = lag(Close, 1),
    Close_Lag7 = lag(Close, 7)
  )

cor(btc$Close, btc$Close_Lag1, use = "complete.obs")
cor(btc$Close, btc$Close_Lag7, use = "complete.obs")


# 6A. Correlation Matrix

cor_matrix <- btc %>%
  select(Open, High, Low, Close, Volume) %>%
  cor(use = "complete.obs")

print(cor_matrix)

# Heatmap
cor_df <- as.data.frame(as.table(cor_matrix))

ggplot(cor_df, aes(Var1, Var2, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = round(Freq, 2)), color = "white") +
  scale_fill_gradient2(low = "red", high = "brown", mid = "white",
                       midpoint = 0, limit = c(-1, 1)) +
  labs(title = "Correlation Matrix Heatmap",
       x = "", y = "") +
  theme_minimal()


# 7. Log transformation and stationarity test

btc$LogClose <- log(btc$Close)
btc_log_ts <- ts(btc$LogClose, frequency = 7)

adf.test(na.omit(btc_log_ts))

diff_log_ts <- diff(btc_log_ts)
adf.test(na.omit(diff_log_ts))


# 8. Train-test split

n <- length(btc_ts)
train_size <- floor(0.8 * n)

train <- window(btc_ts, end = c(1, train_size))
test <- window(btc_ts, start = c(1, train_size + 1))

train_log <- window(btc_log_ts, end = c(1, train_size))


# 9. Baseline models

naive_fit <- naive(train, h = length(test))
snaive_fit <- snaive(train, h = length(test))


# 10. ETS model

ets_fit <- ets(train)
ets_fc <- forecast(ets_fit, h = length(test))


# 11. ARIMA model

arima_fit <- auto.arima(train,
                        seasonal = TRUE,
                        stepwise = FALSE,
                        approximation = FALSE)

summary(arima_fit)
checkresiduals(arima_fit)

arima_fc <- forecast(arima_fit, h = length(test))


# 12. TSLM model

tslm_fit <- tslm(train ~ trend)
tslm_fc <- forecast(tslm_fit, h = length(test))


# 13. Model evaluation

naive_acc <- accuracy(naive_fit, test)
snaive_acc <- accuracy(snaive_fit, test)
ets_acc <- accuracy(ets_fc, test)
arima_acc <- accuracy(arima_fc, test)
tslm_acc <- accuracy(tslm_fc, test)

model_results <- data.frame(
  Model = c("Naive", "Seasonal Naive", "ETS", "ARIMA", "TSLM"),
  
  RMSE = c(naive_acc[2, "RMSE"],
           snaive_acc[2, "RMSE"],
           ets_acc[2, "RMSE"],
           arima_acc[2, "RMSE"],
           tslm_acc[2, "RMSE"]),
  
  MAE = c(naive_acc[2, "MAE"],
          snaive_acc[2, "MAE"],
          ets_acc[2, "MAE"],
          arima_acc[2, "MAE"],
          tslm_acc[2, "MAE"]),
  
  MAPE = c(naive_acc[2, "MAPE"],
           snaive_acc[2, "MAPE"],
           ets_acc[2, "MAPE"],
           arima_acc[2, "MAPE"],
           tslm_acc[2, "MAPE"]),
  
  MASE = c(naive_acc[2, "MASE"],
           snaive_acc[2, "MASE"],
           ets_acc[2, "MASE"],
           arima_acc[2, "MASE"],
           tslm_acc[2, "MASE"])
)

print(model_results)
write.csv(model_results, "model_comparison_results.csv", row.names = FALSE)



# 14. Forecast plot comparison

autoplot(train) +
  autolayer(test, series = "Actual") +
  autolayer(naive_fit$mean, series = "Naive Forecast") +
  autolayer(snaive_fit$mean, series = "Seasonal Naive Forecast") +
  autolayer(ets_fc$mean, series = "ETS Forecast") +
  autolayer(arima_fc$mean, series = "ARIMA Forecast") +
  autolayer(tslm_fc$mean, series = "TSLM Forecast") +
  labs(title = "Bitcoin Forecast Model Comparison",
       x = "Time", y = "Close Price")


# 15. Final model

final_fit <- auto.arima(btc_ts,
                        seasonal = TRUE,
                        stepwise = FALSE,
                        approximation = FALSE)

summary(final_fit)
checkresiduals(final_fit)


# 16. Forecast 30 days

forecast_30 <- forecast(final_fit, h = 30)

autoplot(forecast_30) +
  labs(title = "30-Day Bitcoin Price Forecast",
       x = "Time", y = "Close Price")


# 17. Forecast 365 days

forecast_365 <- forecast(final_fit, h = 365)

autoplot(forecast_365) +
  labs(title = "365-Day Bitcoin Price Forecast",
       x = "Time", y = "Close Price")


# 18. Save forecasts

forecast_30_df <- data.frame(
  Day = 1:30,
  Forecast = as.numeric(forecast_30$mean),
  Lo80 = as.numeric(forecast_30$lower[,1]),
  Hi80 = as.numeric(forecast_30$upper[,1]),
  Lo95 = as.numeric(forecast_30$lower[,2]),
  Hi95 = as.numeric(forecast_30$upper[,2])
)

forecast_365_df <- data.frame(
  Day = 1:365,
  Forecast = as.numeric(forecast_365$mean),
  Lo80 = as.numeric(forecast_365$lower[,1]),
  Hi80 = as.numeric(forecast_365$upper[,1]),
  Lo95 = as.numeric(forecast_365$lower[,2]),
  Hi95 = as.numeric(forecast_365$upper[,2])
)

write.csv(forecast_30_df, "btc_forecast_30day.csv", row.names = FALSE)
write.csv(forecast_365_df, "btc_forecast_365day.csv", row.names = FALSE)


# 19. Residual diagnostics

Acf(residuals(final_fit), main = "ACF of ARIMA Residuals")
Pacf(residuals(final_fit), main = "PACF of ARIMA Residuals")
hist(residuals(final_fit), main = "Residual Histogram", xlab = "Residuals")
qqnorm(residuals(final_fit))
qqline(residuals(final_fit), col = "red")


# 20. Save cleaned data

write.csv(btc, "btc_cleaned_data.csv", row.names = FALSE)