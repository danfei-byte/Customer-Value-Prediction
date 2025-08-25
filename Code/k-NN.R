data <- read.csv("final_data.csv")

install.packages("FNN")
library(FNN)

X <- data[,!(names(data) %in% c("customer_id", "num_purchases", "club_member_status", "fashion_news_frequency", "avg_age", 
              "Shoes & Socks", "Blouses, Tops & Shirts", "Hoodies & Outerwear", "Skirts, Shorts.and.Tights", 
              "Swimwear", "Trousers", "Knitwear",	"T-shirts", "Nightwear",	"Underwear", "Accessories",	
              "Dresses, Jumpsuits.and.Sets"))]


Y <- data$aov

#
set.seed(123)

n <- nrow(data)

train_index <- sample(1:n, size = 0.8*n)

X_train <- X[train_index, ]
Y_train <- Y[train_index]

X_test <- X[-train_index, ]
Y_test <- Y[-train_index]

#
err_by_k <- rep(0,10)

for (k in 1:10){
  knn_model <- knn.reg(train = X_train, test = X_test, y=Y_train, k=k)
  predictions <- knn_model$pred
  err_by_k[k] <- mean(abs(predictions - Y_test))
}

cbind(k = 1:10, mae = err_by_k)

plot(1:10, err_by_k, type = "b", pch = 19, col = "steelblue",
     xlab = "k (Number of Neighbors)", ylab = "MAE",
     main = "k-NN Regression: MAE vs k")
