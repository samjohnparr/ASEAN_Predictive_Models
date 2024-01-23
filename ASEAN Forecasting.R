library(dplyr)
library(echos) 
library(forecast)
library(GreyModel)
library(Metrics)
library(readxl)
library(TSSVM)
library(xgboost)

ASEAN <- read_excel("D:/Data for Research/Energy/World Development Indicators/WDI.xlsx")

Indonesia <- ts(ASEAN$Indonesia, start = 1971, frequency = 1)
Thailand <- ts(ASEAN$Thailand, start = 1971)
Singapore <- ts(ASEAN$Singapore, start = 1971)
Vietnam <- ts(ASEAN$Vietnam, start = 1971)
Malaysia <- ts(ASEAN$Malaysia, start = 1971)
Philippines <- ts(ASEAN$Philippines, start = 1971)
Myanmar <- ts(ASEAN$Myanmar, start = 1971)
Cambodia <- ts(na.omit(ASEAN$Cambodia), start = 1995)
Brunei <- ts(ASEAN$`Brunei Darussalam`, start = 1971)


##########
#  ARIMA #
##########

ARIMA_Indonesia <- auto.arima(Indonesia, ic = "aic", test = "adf", trace = TRUE, stationary = FALSE)
lmtest::coeftest(ARIMA_Indonesia)
checkresiduals(ARIMA_Indonesia)
forecast::accuracy(Indonesia,ARIMA_Indonesia$fitted)

ARIMA_Thailand <- auto.arima(Thailand, ic = "aic", test = "adf", trace = TRUE, stationary = FALSE, allowdrift = FALSE)
lmtest::coeftest(ARIMA_Thailand)
checkresiduals(ARIMA_Thailand)
forecast::accuracy(Thailand,ARIMA_Thailand$fitted)

ARIMA_Singapore <- auto.arima(Singapore, ic = "aic", test = "adf", trace = TRUE, stationary = FALSE, allowdrift = FALSE)
lmtest::coeftest(ARIMA_Singapore)
checkresiduals(ARIMA_Singapore)
forecast::accuracy(Singapore,ARIMA_Singapore$fitted)

ARIMA_Vietnam <- auto.arima(Vietnam, ic = "aic", test = "adf", trace = TRUE, stationary = FALSE, allowdrift = FALSE)
lmtest::coeftest(ARIMA_Vietnam)
checkresiduals(ARIMA_Vietnam)
forecast::accuracy(Vietnam,ARIMA_Vietnam$fitted)

ARIMA_Malaysia <- auto.arima(Malaysia, ic = "aic", test = "adf", trace = TRUE, stationary = FALSE, allowdrift = FALSE)
lmtest::coeftest(ARIMA_Malaysia)
checkresiduals(ARIMA_Malaysia)
forecast::accuracy(Malaysia,ARIMA_Malaysia$fitted)

ARIMA_Philippines <- auto.arima(Philippines, ic = "aic", test = "adf", trace = TRUE, stationary = FALSE, allowdrift = FALSE)
lmtest::coeftest(ARIMA_Philippines)
checkresiduals(ARIMA_Philippines)
forecast::accuracy(Philippines,ARIMA_Philippines$fitted)

ARIMA_Myanmar <- auto.arima(Myanmar, ic = "aic", test = "adf", trace = TRUE, stationary = FALSE, allowdrift = FALSE)
lmtest::coeftest(ARIMA_Myanmar)
checkresiduals(ARIMA_Myanmar)
forecast::accuracy(Myanmar,ARIMA_Myanmar$fitted)

ARIMA_Cambodia <- auto.arima(Cambodia, ic = "aic", test = "adf", trace = TRUE, stationary = FALSE, allowdrift = FALSE)
lmtest::coeftest(ARIMA_Cambodia)
checkresiduals(ARIMA_Cambodia)
forecast::accuracy(na.omit(ASEAN$Cambodia),ARIMA_Cambodia$fitted)

ARIMA_Brunei <- auto.arima(Brunei, ic = "aic", test = "adf", trace = TRUE, stationary = FALSE, allowdrift = FALSE)
lmtest::coeftest(ARIMA_Brunei)
checkresiduals(ARIMA_Brunei)
forecast::accuracy(ASEAN$`Brunei Darussalam`,ARIMA_Brunei$fitted)

##################
#   Grey Model   #
##################

GM_test(Indonesia)
GM_Indonesia <- GM(Indonesia)
GM_Indonesia$a
GM_Indonesia$b
GM_Indonesia$RMSE_Grey
GM_Indonesia$MAPE_Grey

GM_test(Thailand)
GM_Thailand <- GM(Thailand)
GM_Thailand$a
GM_Thailand$b
GM_Thailand$RMSE_Grey
GM_Thailand$MAPE_Grey

GM_test(Singapore)
GM_Singapore <- GM(Singapore)
GM_Singapore$a
GM_Singapore$b
GM_Singapore$RMSE_Grey
GM_Singapore$MAPE_Grey

GM_test(Vietnam)
GM_Vietnam <- GM(Vietnam)
GM_Vietnam$a
GM_Vietnam$b
GM_Vietnam$RMSE_Grey
GM_Vietnam$MAPE_Grey

GM_test(Malaysia)
GM_Malaysia <- GM(Malaysia)
GM_Malaysia$a
GM_Malaysia$b
GM_Malaysia$RMSE_Grey
GM_Malaysia$MAPE_Grey

GM_test(Philippines)
GM_Philippines <- GM(Philippines)
GM_Philippines$a
GM_Philippines$b
GM_Philippines$RMSE_Grey
GM_Philippines$MAPE_Grey

GM_test(Myanmar)
GM_Myanmar <- GM(Myanmar)
GM_Myanmar$a
GM_Myanmar$b
GM_Myanmar$RMSE_Grey
GM_Myanmar$MAPE_Grey

GM_test(Cambodia)
GM_Cambodia <- GM(Cambodia)
GM_Cambodia$a
GM_Cambodia$b
GM_Cambodia$RMSE_Grey
GM_Cambodia$MAPE_Grey

GM_test(Brunei)
GM_Brunei <- GM(Brunei)
GM_Brunei$a
GM_Brunei$b
GM_Brunei$RMSE_Grey
GM_Brunei$MAPE_Grey


###################################
#  Neural Network Autoregressive  #
###################################

NNAR_Indonesia <- nnetar(Indonesia)
summary(NNAR_Indonesia)
forecast::accuracy(ASEAN$Indonesia,NNAR_Indonesia$fitted)

NNAR_Thailand <- nnetar(Thailand)
summary(NNAR_Thailand)
forecast::accuracy(Thailand,NNAR_Thailand$fitted)

NNAR_Singapore <- nnetar(Singapore)
summary(NNAR_Singapore)
forecast::accuracy(Singapore,NNAR_Singapore$fitted)

NNAR_Vietnam <- nnetar(Vietnam)
summary(NNAR_Vietnam)
forecast::accuracy(Vietnam,NNAR_Vietnam$fitted)

NNAR_Malaysia <- nnetar(Malaysia)
summary(NNAR_Malaysia)
forecast::accuracy(ASEAN$Malaysia,NNAR_Malaysia$fitted)

NNAR_Philippines <- nnetar(Philippines)
summary(NNAR_Philippines)
forecast::accuracy(Philippines,NNAR_Philippines$fitted)

NNAR_Myanmar <- nnetar(Myanmar)
summary(NNAR_Myanmar)
forecast::accuracy(Myanmar,NNAR_Myanmar$fitted)

NNAR_Cambodia <- nnetar(Cambodia)
summary(NNAR_Cambodia)
forecast::accuracy(Cambodia,NNAR_Cambodia$fitted)

NNAR_Brunei <- nnetar(Brunei)
summary(NNAR_Brunei)
forecast::accuracy(Brunei,NNAR_Brunei$fitted)

#########
#  ESN  #
#########

ESN_Indonesia <- train_esn(y = as.numeric(Indonesia))
ESN_Indonesia$method$model_layers
rmse(na.omit(ESN_Indonesia$actual),na.omit(ESN_Indonesia$fitted))
mape(na.omit(ESN_Indonesia$actual),na.omit(ESN_Indonesia$fitted))*100

#ts.plot(ts(ESN_Indonesia$actual),ts(ESN_Indonesia$fitted), col = c("blue","red"), type = "l")

ESN_Thailand <- train_esn(y = as.numeric(Thailand))
ESN_Thailand$method$model_layers
rmse(na.omit(ESN_Thailand$actual),na.omit(ESN_Thailand$fitted))
mape(na.omit(ESN_Thailand$actual),na.omit(ESN_Thailand$fitted))*100


ESN_Singapore <- train_esn(y = as.numeric(Singapore))
ESN_Singapore$method$model_layers
rmse(na.omit(ESN_Singapore$actual),na.omit(ESN_Singapore$fitted))
mape(na.omit(ESN_Singapore$actual),na.omit(ESN_Singapore$fitted))*100

ESN_Vietnam <- train_esn(y = as.numeric(Vietnam))
ESN_Vietnam$method$model_layers
rmse(na.omit(ESN_Vietnam$actual),na.omit(ESN_Vietnam$fitted))
mape(na.omit(ESN_Vietnam$actual),na.omit(ESN_Vietnam$fitted))*100

ESN_Malaysia <- train_esn(y = as.numeric(Malaysia))
ESN_Malaysia$method$model_layers
rmse(na.omit(ESN_Malaysia$actual),na.omit(ESN_Malaysia$fitted))
mape(na.omit(ESN_Malaysia$actual),na.omit(ESN_Malaysia$fitted))*100

ESN_Philippines <- train_esn(y = as.numeric(Philippines))
ESN_Philippines$method$model_layers
rmse(na.omit(ESN_Philippines$actual),na.omit(ESN_Philippines$fitted))
mape(na.omit(ESN_Philippines$actual),na.omit(ESN_Philippines$fitted))*100

ESN_Myanmar <- train_esn(y = as.numeric(Myanmar))
ESN_Myanmar$method$model_layers
rmse(na.omit(ESN_Myanmar$actual),na.omit(ESN_Myanmar$fitted))
mape(na.omit(ESN_Myanmar$actual),na.omit(ESN_Myanmar$fitted))*100

ESN_Cambodia <- train_esn(y = as.numeric(Cambodia))
ESN_Cambodia$method$model_layers
rmse(na.omit(ESN_Cambodia$actual),na.omit(ESN_Cambodia$fitted))
mape(na.omit(ESN_Cambodia$actual),na.omit(ESN_Cambodia$fitted))*100

ESN_Brunei <- train_esn(y = as.numeric(Brunei))
ESN_Brunei$method$model_layers
rmse(na.omit(ESN_Brunei$actual),na.omit(ESN_Brunei$fitted))
mape(na.omit(ESN_Brunei$actual),na.omit(ESN_Brunei$fitted))*100

#ts.plot(ts(ESN_Brunei$actual),ts(ESN_Brunei$fitted), col = c("blue","red"), type = "l")

##############
#   AR-SVM   #
##############

ARSVM_Indonesia <- ARSVM(Indonesia, h = 20)
ARSVM_Indonesia$`Model Summary`
ARSVM_Indonesia$RMSE
ARSVM_Indonesia$MAPE

ARSVM_Thailand <- ARSVM(Thailand, h = 20)
ARSVM_Thailand$`Model Summary`
ARSVM_Thailand$RMSE
ARSVM_Thailand$MAPE

ARSVM_Singapore <- ARSVM(Singapore, h = 20)
ARSVM_Singapore$`Model Summary`
ARSVM_Singapore$RMSE
ARSVM_Singapore$MAPE

ARSVM_Vietnam <- ARSVM(Vietnam, h = 20)
ARSVM_Vietnam$`Model Summary`
ARSVM_Vietnam$RMSE
ARSVM_Vietnam$MAPE

ARSVM_Malaysia <- ARSVM(Malaysia, h = 20)
ARSVM_Malaysia$`Model Summary`
ARSVM_Malaysia$RMSE
ARSVM_Malaysia$MAPE

ARSVM_Philippines <- ARSVM(Philippines, h = 20)
ARSVM_Philippines$`Model Summary`
ARSVM_Philippines$RMSE
ARSVM_Philippines$MAPE

ARSVM_Myanmar <- ARSVM(Myanmar, h = 20)
ARSVM_Myanmar$`Model Summary`
ARSVM_Myanmar$RMSE
ARSVM_Myanmar$MAPE

ARSVM_Cambodia <- ARSVM(Cambodia, h = 20)
ARSVM_Cambodia$`Model Summary`
ARSVM_Cambodia$RMSE
ARSVM_Cambodia$MAPE

ARSVM_Brunei <- ARSVM(Brunei, h = 20)
ARSVM_Brunei$`Model Summary`
ARSVM_Brunei$RMSE
ARSVM_Brunei$MAPE

###############
#   XGBoost   #
###############

create_lagged_features <- function(ts_data, lag = 1) {
  data.frame(
    Y = ts_data,
    lag = stats::lag(ts_data, -lag)
  )
}

params <- list(
  objective = "reg:squarederror",
  eta = 0.1,
  max_depth = 3,
  nthread = 2
)

predict_next_values <- function(model, last_value, n_ahead) {
  predictions = numeric(n_ahead)
  current_input = last_value
  
  for (i in 1:n_ahead) {
    # Predict the next value
    next_value = predict(model, as.matrix(current_input))
    
    # Store the prediction
    predictions[i] = next_value
    
    # Update the input for the next prediction
    current_input = next_value
  }
  
  return(predictions)
}

#Indonesia

Indonesia_Lag <- create_lagged_features(Indonesia, lag = 1) %>% na.omit()
Indonesia_train_data <- as.matrix(Indonesia_Lag[,-1])
Indonesia_train_label <- Indonesia_Lag$Y

XGB_Indonesia <- xgboost(data = Indonesia_train_data, label = Indonesia_train_label, nrounds = 100, params = params)

XGB_Indonesia_predictions <- predict(XGB_Indonesia, Indonesia_train_data)
rmse(Indonesia_train_label, XGB_Indonesia_predictions)
mape(Indonesia_train_label, XGB_Indonesia_predictions)


Indo_last_known_value <- tail(Indonesia_train_label, 1)
predict_next_values(XGB_Indonesia,Indo_last_known_value,20)

#Thailand

Thailand_Lag <- create_lagged_features(Thailand, lag = 1) %>% na.omit()
Thailand_train_data <- as.matrix(Thailand_Lag[,-1])
Thailand_train_label <- Thailand_Lag$Y

XGB_Thailand <- xgboost(data = Thailand_train_data, label = Thailand_train_label, nrounds = 100, params = params)

XGB_Thailand_predictions <- predict(XGB_Thailand, Thailand_train_data)
rmse(Thailand_train_label, XGB_Thailand_predictions)
mape(Thailand_train_label, XGB_Thailand_predictions)


Thailand_last_known_value <- tail(Thailand_train_label, 1)
predict_next_values(XGB_Thailand,Thailand_last_known_value,20)

#Singapore

Singapore_Lag <- create_lagged_features(Singapore, lag = 1) %>% na.omit()
Singapore_train_data <- as.matrix(Singapore_Lag[,-1])
Singapore_train_label <- Singapore_Lag$Y

XGB_Singapore <- xgboost(data = Singapore_train_data, label = Singapore_train_label, nrounds = 100, params = params)

XGB_Singapore_predictions <- predict(XGB_Singapore, Singapore_train_data)
rmse(Singapore_train_label, XGB_Singapore_predictions)
mape(Singapore_train_label, XGB_Singapore_predictions)


Singapore_last_known_value <- tail(Singapore_train_label, 1)
predict_next_values(XGB_Singapore,Singapore_last_known_value,20)

#Vietnam

Vietnam_Lag <- create_lagged_features(Vietnam, lag = 1) %>% na.omit()
Vietnam_train_data <- as.matrix(Vietnam_Lag[,-1])
Vietnam_train_label <- Vietnam_Lag$Y

XGB_Vietnam <- xgboost(data = Vietnam_train_data, label = Vietnam_train_label, nrounds = 100, params = params)

XGB_Vietnam_predictions <- predict(XGB_Vietnam, Vietnam_train_data)
rmse(Vietnam_train_label, XGB_Vietnam_predictions)
mape(Vietnam_train_label, XGB_Vietnam_predictions)


Vietnam_last_known_value <- tail(Vietnam_train_label, 1)
predict_next_values(XGB_Vietnam,Vietnam_last_known_value,20)

#Malaysia

Malaysia_Lag <- create_lagged_features(Malaysia, lag = 1) %>% na.omit()
Malaysia_train_data <- as.matrix(Malaysia_Lag[,-1])
Malaysia_train_label <- Malaysia_Lag$Y

XGB_Malaysia <- xgboost(data = Malaysia_train_data, label = Malaysia_train_label, nrounds = 100, params = params)

XGB_Malaysia_predictions <- predict(XGB_Malaysia, Malaysia_train_data)
rmse(Malaysia_train_label, XGB_Malaysia_predictions)
mape(Malaysia_train_label, XGB_Malaysia_predictions)


Malaysia_last_known_value <- tail(Malaysia_train_label, 1)
predict_next_values(XGB_Malaysia,Malaysia_last_known_value,20)

#Philippines

Philippines_Lag <- create_lagged_features(Philippines, lag = 1) %>% na.omit()
Philippines_train_data <- as.matrix(Philippines_Lag[,-1])
Philippines_train_label <- Philippines_Lag$Y

XGB_Philippines <- xgboost(data = Philippines_train_data, label = Philippines_train_label, nrounds = 100, params = params)

XGB_Philippines_predictions <- predict(XGB_Philippines, Philippines_train_data)
rmse(Philippines_train_label, XGB_Philippines_predictions)
mape(Philippines_train_label, XGB_Philippines_predictions)


Philippines_last_known_value <- tail(Philippines_train_label, 1)
predict_next_values(XGB_Philippines,Philippines_last_known_value,20)

#Myanmar

Myanmar_Lag <- create_lagged_features(Myanmar, lag = 1) %>% na.omit()
Myanmar_train_data <- as.matrix(Myanmar_Lag[,-1])
Myanmar_train_label <- Myanmar_Lag$Y

XGB_Myanmar <- xgboost(data = Myanmar_train_data, label = Myanmar_train_label, nrounds = 100, params = params)

XGB_Myanmar_predictions <- predict(XGB_Myanmar, Myanmar_train_data)
rmse(Myanmar_train_label, XGB_Myanmar_predictions)
mape(Myanmar_train_label, XGB_Myanmar_predictions)


Myanmar_last_known_value <- tail(Myanmar_train_label, 1)
predict_next_values(XGB_Myanmar,Myanmar_last_known_value,20)

#Cambodia

Cambodia_Lag <- create_lagged_features(Cambodia, lag = 1) %>% na.omit()
Cambodia_train_data <- as.matrix(Cambodia_Lag[,-1])
Cambodia_train_label <- Cambodia_Lag$Y

XGB_Cambodia <- xgboost(data = Cambodia_train_data, label = Cambodia_train_label, nrounds = 100, params = params)

XGB_Cambodia_predictions <- predict(XGB_Cambodia, Cambodia_train_data)
rmse(Cambodia_train_label, XGB_Cambodia_predictions)
mape(Cambodia_train_label, XGB_Cambodia_predictions)

Cambodia_last_known_value <- tail(Cambodia_train_label, 1)
predict_next_values(XGB_Cambodia, Cambodia_last_known_value, 20)

#Brunei

Brunei_Lag <- create_lagged_features(Brunei, lag = 1) %>% na.omit()
Brunei_train_data <- as.matrix(Brunei_Lag[,-1])
Brunei_train_label <- Brunei_Lag$Y

XGB_Brunei <- xgboost(data = Brunei_train_data, label = Brunei_train_label, nrounds = 100, params = params)

XGB_Brunei_predictions <- predict(XGB_Brunei, Brunei_train_data)
rmse(Brunei_train_label, XGB_Brunei_predictions)
mape(Brunei_train_label, XGB_Brunei_predictions)

Brunei_last_known_value <- tail(Brunei_train_label, 1)
predict_next_values(XGB_Brunei, Brunei_last_known_value, 20)

################################################################################

#Brunei

ts.plot(Brunei, ARIMA_Brunei$fitted, GM_Brunei$fitted, NNAR_Brunei$fitted, ESN_Brunei$fitted, c(ARSVM_Brunei$fitted,NA), ts(XGB_Brunei_predictions, start = 1971), 
        col = c("black","blue","red","green","orange","purple","brown"), xlab = "Year", 
        ylab = "Electric Power Consumption (kWh)", main = "Brunei Darussalam", lwd = 2)
legend("topleft",bty="o",lty=c(1,1),col=c("black","blue","red","green","orange","purple","brown"),
      legend=c("Actual","ARIMA","GM","NNAR","ESN","ARSVM","XGBoost"),cex=0.7,inset=0.025,lwd=2)

#Cambodia

ts.plot(Cambodia, ARIMA_Cambodia$fitted, GM_Cambodia$fitted, NNAR_Cambodia$fitted, ESN_Cambodia$fitted, c(ARSVM_Cambodia$fitted,NA), ts(XGB_Cambodia_predictions, start = 1995), 
        col = c("black","blue","red","green","orange","purple","brown"), xlab = "Year", 
        ylab = "Electric Power Consumption (kWh)", main = "Cambodia", lwd = 2)
legend("topleft",bty="o",lty=c(1,1),col=c("black","blue","red","green","orange","purple","brown"),
       legend=c("Actual","ARIMA","GM","NNAR","ESN","ARSVM","XGBoost"),cex=0.7,inset=0.025,lwd=2)

#Indonesia

ts.plot(Indonesia, ARIMA_Indonesia$fitted, GM_Indonesia$fitted, NNAR_Indonesia$fitted, ESN_Indonesia$fitted, c(ARSVM_Indonesia$fitted,NA), ts(XGB_Indonesia_predictions, start = 1971), 
        col = c("black","blue","red","green","orange","purple","brown"), xlab = "Year", 
        ylab = "Electric Power Consumption (kWh)", main = "Indonesia", lwd = 2)
legend("topleft",bty="o",lty=c(1,1),col=c("black","blue","red","green","orange","purple","brown"),
       legend=c("Actual","ARIMA","GM","NNAR","ESN","ARSVM","XGBoost"),cex=0.7,inset=0.025,lwd=2)

#Malaysia

ts.plot(Malaysia, ARIMA_Malaysia$fitted, GM_Malaysia$fitted, NNAR_Malaysia$fitted, ESN_Malaysia$fitted, c(ARSVM_Malaysia$fitted,NA), ts(XGB_Malaysia_predictions, start = 1971), 
        col = c("black","blue","red","green","orange","purple","brown"), xlab = "Year", 
        ylab = "Electric Power Consumption (kWh)", main = "Malaysia", lwd = 2)
legend("topleft",bty="o",lty=c(1,1),col=c("black","blue","red","green","orange","purple","brown"),
       legend=c("Actual","ARIMA","GM","NNAR","ESN","ARSVM","XGBoost"),cex=0.7,inset=0.025,lwd=2)

#Myanmar

ts.plot(Myanmar, ARIMA_Myanmar$fitted, GM_Myanmar$fitted, NNAR_Myanmar$fitted, ESN_Myanmar$fitted, c(ARSVM_Myanmar$fitted,NA), ts(XGB_Myanmar_predictions, start = 1971), 
        col = c("black","blue","red","green","orange","purple","brown"), xlab = "Year", 
        ylab = "Electric Power Consumption (kWh)", main = "Myanmar", lwd = 2)
legend("topleft",bty="o",lty=c(1,1),col=c("black","blue","red","green","orange","purple","brown"),
       legend=c("Actual","ARIMA","GM","NNAR","ESN","ARSVM","XGBoost"),cex=0.7,inset=0.025,lwd=2)

#Philippines

ts.plot(Philippines, ARIMA_Philippines$fitted, GM_Philippines$fitted, NNAR_Philippines$fitted, ESN_Philippines$fitted, c(ARSVM_Philippines$fitted,NA), ts(XGB_Philippines_predictions, start = 1971), 
        col = c("black","blue","red","green","orange","purple","brown"), xlab = "Year", 
        ylab = "Electric Power Consumption (kWh)", main = "Philippines", lwd = 2)
legend("topleft",bty="o",lty=c(1,1),col=c("black","blue","red","green","orange","purple","brown"),
       legend=c("Actual","ARIMA","GM","NNAR","ESN","ARSVM","XGBoost"),cex=0.7,inset=0.025,lwd=2)

#Singapore

ts.plot(Singapore, ARIMA_Singapore$fitted, GM_Singapore$fitted, NNAR_Singapore$fitted, ESN_Singapore$fitted, c(ARSVM_Singapore$fitted,NA), ts(XGB_Singapore_predictions, start = 1971), 
        col = c("black","blue","red","green","orange","purple","brown"), xlab = "Year", 
        ylab = "Electric Power Consumption (kWh)", main = "Singapore", lwd = 2)
legend("topleft",bty="o",lty=c(1,1),col=c("black","blue","red","green","orange","purple","brown"),
       legend=c("Actual","ARIMA","GM","NNAR","ESN","ARSVM","XGBoost"),cex=0.7,inset=0.025,lwd=2)

#Thailand

ts.plot(Thailand, ARIMA_Thailand$fitted, GM_Thailand$fitted, NNAR_Thailand$fitted, ESN_Thailand$fitted, c(ARSVM_Thailand$fitted,NA), ts(XGB_Thailand_predictions, start = 1971), 
        col = c("black","blue","red","green","orange","purple","brown"), xlab = "Year", 
        ylab = "Electric Power Consumption (kWh)", main = "Thailand", lwd = 2)
legend("topleft",bty="o",lty=c(1,1),col=c("black","blue","red","green","orange","purple","brown"),
       legend=c("Actual","ARIMA","GM","NNAR","ESN","ARSVM","XGBoost"),cex=0.7,inset=0.025,lwd=2)

#Vietnam

ts.plot(Vietnam, ARIMA_Vietnam$fitted, GM_Vietnam$fitted, NNAR_Vietnam$fitted, ESN_Vietnam$fitted, c(ARSVM_Vietnam$fitted,NA), ts(XGB_Vietnam_predictions, start = 1971), 
        col = c("black","blue","red","green","orange","purple","brown"), xlab = "Year", 
        ylab = "Electric Power Consumption (kWh)", main = "Vietnam", lwd = 2)
legend("topleft",bty="o",lty=c(1,1),col=c("black","blue","red","green","orange","purple","brown"),
       legend=c("Actual","ARIMA","GM","NNAR","ESN","ARSVM","XGBoost"),cex=0.7,inset=0.025,lwd=2)
