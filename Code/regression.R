setwd("/Users/dynamite/Desktop/DA/Projects/ML_H&M")
data <- read.csv("final_data.csv")

library(leaps)
library(lars)
library(Metrics)
library(caret)

# Split data
set.seed(746)
train_ind <- sample(seq_len(nrow(data)), size=0.8*nrow(data))
data_train <- data[train_ind, ]
data_test <- data[-train_ind, ]

colnames(data_train)

data_train$club_member_status <- as.factor(data_train$club_member_status)
data_train$fashion_news_frequency <- as.factor(data_train$fashion_news_frequency)

# forward step-wise
y <- data_train$aov
x <- cbind(data_train$num_purchases, data_train$club_member_status, data_train$fashion_news_frequency, data_train$avg_age,data_train$Shoes...Socks, 
           data_train$Blouses..Tops...Shirts, data_train$Hoodies...Outerwear, data_train$Skirts..Shorts...Tights, data_train$Swimwear, data_train$Trousers, data_train$Knitwear,
           data_train$T.shirts, data_train$Nightwear, data_train$Underwear, data_train$Accessories, data_train$Dresses..Jumpsuits...Sets)
res <- lars(x, y, type="stepwise")
print(summary(res)) 
res # all Xs should be chosen
plot(res, plottype = "Cp")

# Fit final linear model
final_lm <- lm(aov ~ num_purchases + club_member_status + fashion_news_frequency + avg_age +
                 Shoes...Socks + Blouses..Tops...Shirts + Hoodies...Outerwear +
                 Skirts..Shorts...Tights + Swimwear + Trousers + Knitwear +
                 T.shirts + Nightwear + Underwear + Accessories + Dresses..Jumpsuits...Sets,data = data_train)
summary(final_lm)
# Predict on test set
pred_lm <- predict(final_lm, newdata = data_test)

rmse(actual = data_test$aov, pred = pred_lm) # RMSE = 0.01146997
mae(actual = data_test$aov, pred = pred_lm) # MAE = 0.008111659
R2(pred_lm, data_test$aov) # R2 = 0.1432302

