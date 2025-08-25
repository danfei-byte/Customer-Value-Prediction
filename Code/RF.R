setwd("/Users/dynamite/Desktop/DA/Projects/ML_H&M")
data <- read.csv("final_data.csv")
View(data)
str(data)
summary(data)

colSums(is.na(data)) # check missing values

library(tidyverse)
library(caret)
library(randomForest)
library(Metrics)
##### Task:
# The Y variable should be the average order revenue per customer (so AOV per customer) to predict next year's AOV per customer

# Convert to factors
data$club_member_status <- as.factor(data$club_member_status)
data$fashion_news_frequency <- as.factor(data$fashion_news_frequency)

 # Split data
set.seed(746)
train_ind <- sample(seq_len(nrow(data)), size=0.8*nrow(data))
data_train <- data[train_ind, ]
data_test <- data[-train_ind, ]

colnames(data_train)

########################
# Fit model, set ntree=100
# mtry = default: Number of features/3
########################
 set.seed(746)
 rfitt <- randomForest(aov ~ num_purchases + club_member_status + fashion_news_frequency + avg_age +
                         Shoes...Socks + Blouses..Tops...Shirts + Hoodies...Outerwear +
                         Skirts..Shorts...Tights + Swimwear + Trousers + Knitwear +
                         T.shirts + Nightwear + Underwear + Accessories + Dresses..Jumpsuits...Sets,
                       data = data_train,
                       importance=TRUE, 
                       na.action=na.omit,
                       ntree=100
                      )
plot(rfitt) # OOB error decreases sharply at the beginning and then levels off around 50–60 trees
print(rfitt) # MSE: 0.000117, R2=25.42, based on OBB error
# OOB error is a type of internal cross-validation used in Random Forest:
# Each tree is trained on a bootstrap sample (around 63% of the data),
# The remaining ~37% is left out (“out-of-bag”) for testing,
# The model averages prediction error over these OOB predictions.

importance(rfitt)
varImpPlot(rfitt) # Variable importance
 
pred <- predict(rfitt, data_train) # predict on training data
actual <- data_train$aov
rmse_train <- rmse(actual, pred)
r2_train <- R2(pred, actual)
rmse_train # 0.009987
r2_train # 0.3634768

pred_rf <- predict(rfitt, data_test) # predict on testing data
actual <- data_test$aov
 # Calculate RMSE
 rmse_rf <- rmse(actual, pred_rf)
 # Calculate R²
 r2_rf <- R2(pred_rf, actual)
 cat("Test RMSE:", rmse_rf, "\n") # RMSE = 0.01076063, on average, this model’s predictions of AOV are off by about 0.1056 units
 cat("Test R²:", r2_rf, "\n") # R^2 = 0.2467489, model explains ~25.89% of the variance in the aov values on the testing set

 ########################
 # Fit model, set ntree=60
 ########################
set.seed(746)
rfitt_60 <- randomForest(aov ~ num_purchases + club_member_status + fashion_news_frequency + avg_age +
                         Shoes...Socks + Blouses..Tops...Shirts + Hoodies...Outerwear +
                         Skirts..Shorts...Tights + Swimwear + Trousers + Knitwear +
                         T.shirts + Nightwear + Underwear + Accessories + Dresses..Jumpsuits...Sets,
                       data = data_train,
                       importance=TRUE, 
                       na.action=na.omit,
                       ntree=60
 )
plot(rfitt_60) 
print(rfitt_60) # MSE = 0.0001165533, R2 = 25.29

importance(rfitt_60)
varImpPlot(rfitt_60) # Variable importance

pred_60 <- predict(rfitt_60, data_train) # predict on training data
actual <- data_train$aov
rmse_train <- rmse(actual, pred_60)
r2_train <- R2(pred_60, actual)
rmse_train # 0.009986451
r2_train # 0.36473

pred_rf <- predict(rfitt_60, data_test) # predict on testing data
actual <- data_test$aov
# Calculate RMSE
rmse_rf <- rmse(actual, pred_rf)
# Calculate R²
r2_rf <- R2(pred_rf, actual)
cat("Test RMSE:", rmse_rf, "\n") # RMSE = 0.01076448
cat("Test R²:", r2_rf, "\n") # Test R2 = 0.246049

# Summary: when ntree=100, the model performs slight better, no huge difference
# it's likely that model performance is stable across 60 and 100 trees


##############################
# Fit model, ntree=100, use caret to tune mrty
###############################
# Set up training control
ctrl <- trainControl(method = "cv", number = 5)

# Train random forest model with tuning mtry
set.seed(746)
tuned_rf_100 <- train(
  aov ~ num_purchases + club_member_status + fashion_news_frequency + avg_age +
    Shoes...Socks + Blouses..Tops...Shirts + Hoodies...Outerwear +
    Skirts..Shorts...Tights + Swimwear + Trousers + Knitwear +
    T.shirts + Nightwear + Underwear + Accessories + Dresses..Jumpsuits...Sets,
  data = data_train,
  method = "rf",
  trControl = ctrl,
  tuneGrid = expand.grid(mtry = 2:6),  # Try mtry values from 2 to 6
  ntree = 100
)

# View results
print(tuned_rf_100) # Based on RMSE, optimal mtry = 4
plot(tuned_rf_100) 

plot(varImp(tuned_rf_100)) # Varable importance

pred_rf <- predict(tuned_rf_100, data_test) # predict on testing data
actual <- data_test$aov
# Calculate RMSE
rmse_rf <- rmse(actual, pred_rf)
# Calculate R²
r2_rf <- R2(pred_rf, actual)
cat("Test RMSE:", rmse_rf, "\n") # RMSE = 0.0107037
cat("Test R²:", r2_rf, "\n") # Test R2 = 0.2534079

##############################
# Fit model, ntree=60, use caret to tune mrty
###############################
# Set up training control
ctrl <- trainControl(method = "cv", number = 5) # 5-fold CV

# Train random forest model with tuning mtry
set.seed(746)
tuned_rf_60 <- train(
  aov ~ num_purchases + club_member_status + fashion_news_frequency + avg_age +
    Shoes...Socks + Blouses..Tops...Shirts + Hoodies...Outerwear +
    Skirts..Shorts...Tights + Swimwear + Trousers + Knitwear +
    T.shirts + Nightwear + Underwear + Accessories + Dresses..Jumpsuits...Sets,
  data = data_train,
  method = "rf",
  trControl = ctrl,
  tuneGrid = expand.grid(mtry = 2:6),  # Try mtry values from 2 to 6
  ntree = 60
)

# View results
print(tuned_rf_60) # Based on RMSE, optimal mtry = 3, RMSE(min) = 0.01073915 
plot(tuned_rf_60) 

plot(varImp(tuned_rf_60)) # Varable importance

pred_rf <- predict(tuned_rf_60, data_test) # predict on testing data
actual <- data_test$aov
# Calculate RMSE
rmse_rf <- rmse(actual, pred_rf)
# Calculate R²
r2_rf <- R2(pred_rf, actual)
cat("Test RMSE:", rmse_rf, "\n") # RMSE = 0.01070686
cat("Test R²:", r2_rf, "\n") # Test R2 = 0.2544073

