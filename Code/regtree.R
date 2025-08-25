library(rpart)

setwd("C:/Users/Dell/Desktop/UMD/Spring 2025/BUMK 746 Data Science for Customer Analytics/Group Project")
data <- read.csv("final_data.csv")

set.seed(746)
train_ind <- sample(seq_len(nrow(data)), size=0.8*nrow(data))
train <- data[train_ind, ]
test <- data[-train_ind, ]

################################################################################
# REGRESSION TREE
################################################################################

# fitting a tree at an arbitrary cp
regtree <- rpart(aov ~ ., data=train[,-1], method='anova', control=rpart.control(cp=0.0005))
print(regtree)
plot(regtree, uniform = TRUE, main = "Regression Tree")
text(regtree, cex = 0.8, digits=2)

# pruning the tree
printcp(regtree) #xerror is minimized to 0.75355 at cp=0.00071458
plotcp(regtree)

regtree_pruned <- prune(regtree, cp=0.00071458)
plot(regtree_pruned, uniform=TRUE)
text(regtree_pruned)
sort(regtree_pruned$variable.importance, decreasing=TRUE)

# performance on training data
pred_train <- predict(regtree_pruned, train)
rmse_train <- sqrt(mean((train$aov - pred_train)^2))
rss_train <- sum((train$aov - pred_train)^2)
tss_train <- sum((train$aov - mean(train$aov))^2)
r2_train <- 1 - (rss_train / tss_train)

# performance on testing data
pred_test <- predict(regtree_pruned, test)
rmse_test <- sqrt(mean((test$aov - pred_test)^2))
rss_test <- sum((test$aov - pred_test)^2)
tss_test <- sum((test$aov - mean(test$aov))^2)
r2_test <- 1 - (rss_test / tss_test)