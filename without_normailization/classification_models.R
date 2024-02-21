# 6. Classification
# Separating features and target in test, training, and validation sets
X_train <- train_set[, !names(heart) %in% c('target')]
y_train <- train_set$target
X_test <- test_set[, !names(heart) %in% c('target')]
y_test <- test_set$target
X_validation <- validation_set[, !names(heart) %in% c('target')]
y_validation <- validation_set$target

# 6.1 kNN
# 6.1.1 Validation Set
# Training and evaluating the kNN model with different k values
find_best_k <- function(k_values, X_train, y_train, X_validation, y_validation) {
  results <- data.frame(k = numeric(), accuracy = numeric())
  for (k in k_values) {
    model <- knn(train = X_train, test = X_validation, cl = y_train, k = k)
    accuracy <- sum(model == y_validation) / length(y_validation)
    results <- rbind(results, data.frame(k = k, accuracy = accuracy))
  }
  return(results)
}
# Defining the k range to test
k_values <- seq(1, 30, by = 2)
# Initializing variables
best_k <- NULL
best_accuracy <- 0
# Finding the best k
results <- find_best_k(k_values, X_train, y_train, X_validation, y_validation)
# Printing the results
print(results)
# Finding the model with the highest accuracy
best_k_index <- which.max(results$accuracy)
best_accuracy <- results$accuracy[best_k_index]
# Training kNN with the best k
kNN_model <- knn(train = X_train, test = X_validation, cl = y_train, k = best_k_index)
print(kNN_model)
# 7.1.1 Evaluating the model
conf_matrix <- table(kNN_model, test_set$target)
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)
# 8.1.1 Confusion matrix
cat("Confusion Matrix:\n", conf_matrix, "\n\n")
# Printing the evaluation metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
# 9.1.1 Visualizing classifier results
# ROC Curve
roc_curve <- roc(test_set$target, as.numeric(kNN_model), levels = c(0, 1))
plot(roc_curve, col = "blue", main = "ROC Curve", col.main = "darkblue")
# Precision-Recall Curve
pr_curve <- pr.curve(scores.class0 = as.numeric(kNN_model), weights.class0 = ifelse(test_set$target == 0, 1, 0), curve = TRUE)
plot(pr_curve, col = "red", main = "Precision-Recall Curve", col.main = "darkred")

# 6.1.2 Test Set
# Training and evaluating the kNN model with different k values
find_best_k <- function(k_values, X_train, y_train, X_test, y_test) {
  results <- data.frame(k = numeric(), accuracy = numeric())
  for (k in k_values) {
    model <- knn(train = X_train, test = X_test, cl = y_train, k = k)
    accuracy <- sum(model == y_test) / length(y_test)
    results <- rbind(results, data.frame(k = k, accuracy = accuracy))
  }
  return(results)
}
# Defining the k range to test
k_values <- seq(1, 30, by = 2)
# Initializing variables
best_k <- NULL
best_accuracy <- 0
# Finding the best k
results <- find_best_k(k_values, X_train, y_train, X_test, y_test)
# Printing the results
print(results)
# Finding the model with the highest accuracy
best_k_index <- which.max(results$accuracy)
best_accuracy <- results$accuracy[best_k_index]
# Training kNN with the best k
kNN_model <- knn(train = X_train, test = X_test, cl = y_train, k = best_k_index)
print(kNN_model)
# 7.1.2 Evaluating the model
conf_matrix <- table(kNN_model, test_set$target)
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)
# 8.1.2 Confusion matrix
cat("Confusion Matrix:\n", conf_matrix, "\n\n")
# Printing the evaluation metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
# 9.1.2 Visualizing classifier results
# ROC Curve
roc_curve <- roc(test_set$target, as.numeric(kNN_model), levels = c(0, 1))
plot(roc_curve, col = "blue", main = "ROC Curve", col.main = "darkblue")
# Precision-Recall Curve
pr_curve <- pr.curve(scores.class0 = as.numeric(kNN_model), weights.class0 = ifelse(test_set$target == 0, 1, 0), curve = TRUE)
plot(pr_curve, col = "red", main = "Precision-Recall Curve", col.main = "darkred")


# 6.2 SVM
# 6.2.1 Validation Set
# Training SVM model
svm_model <- svm(target ~ ., data = train_set, kernel = "linear")
# Making predictions
svm_predictions <- predict(svm_model, validation_set)
# Converting predictions to binary with a threshold of 0.5
svm_predictions_binary <- ifelse(svm_predictions >= 0.5, 1, 0)
# 7.2.1 Evaluating the model
conf_matrix <- table(Actual = test_set$target, Predicted = svm_predictions_binary)
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)
# 8.2.1 Confusion matrix
cat("Confusion Matrix:\n", conf_matrix, "\n\n")
# Printing the evaluation metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
# 9.2.1 Visualizing classifier results
# ROC Curve for SVM
roc_curve_svm <- roc(test_set$target, svm_predictions, levels = c(0, 1))
plot(roc_curve_svm, col = "blue", main = "ROC Curve (SVM)", col.main = "darkblue")
# Precision-Recall Curve for SVM
pr_curve_svm <- pr.curve(scores.class0 = svm_predictions, weights.class0 = ifelse(test_set$target == 0, 1, 0), curve = TRUE)
plot(pr_curve_svm, col = "red", main = "Precision-Recall Curve (SVM)", col.main = "darkred")

# 6.2.2 Test Set
# Training SVM model
svm_model <- svm(target ~ ., data = train_set, kernel = "linear")
# Making predictions
svm_predictions <- predict(svm_model, test_set)
# Converting predictions to binary with a threshold of 0.5
svm_predictions_binary <- ifelse(svm_predictions >= 0.5, 1, 0)
# 7.2.2 Evaluating the model
conf_matrix <- table(Actual = test_set$target, Predicted = svm_predictions_binary)
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)
# 8.2.2 Confusion matrix
cat("Confusion Matrix:\n", conf_matrix, "\n\n")
# Printing the evaluation metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
# 9.2.2 Visualizing classifier results
# ROC Curve for SVM
roc_curve_svm <- roc(test_set$target, svm_predictions, levels = c(0, 1))
plot(roc_curve_svm, col = "blue", main = "ROC Curve (SVM)", col.main = "darkblue")
# Precision-Recall Curve for SVM
pr_curve_svm <- pr.curve(scores.class0 = svm_predictions, weights.class0 = ifelse(test_set$target == 0, 1, 0), curve = TRUE)
plot(pr_curve_svm, col = "red", main = "Precision-Recall Curve (SVM)", col.main = "darkred")


# 6.3 Decision Tree
# 6.3.1 Validation Set
# Training the model
tree_model <- rpart(target ~ ., data = train_set, method = "class")
# Making predictions
tree_validation_predictions <- predict(tree_model, validation_set, type = "class")
# 7.3.1 Evaluating the model
conf_matrix <-  table(Actual = test_set$target, Predicted = tree_validation_predictions)
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)
# 8.3.1 Confusion matrix
cat("Confusion Matrix:\n", conf_matrix, "\n\n")
# Printing the evaluation metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
# 9.3.1 Visualizing classifier results
# Convert factor predictions to numeric
tree_predictions_numeric <- as.numeric(as.character(tree_validation_predictions))
# ROC Curve for Decision Tree
roc_curve_tree <- roc(test_set$target, tree_predictions_numeric, levels = c(0, 1))
plot(roc_curve_tree, col = "blue", main = "ROC Curve (Decision Tree)", col.main = "darkblue")
# Precision-Recall Curve 
pr_curve_tree <- pr.curve(scores.class0 = tree_validation_predictions, weights.class0 = ifelse(test_set$target == 0, 1, 0), curve = TRUE)
plot(pr_curve_tree, col = "red", main = "Precision-Recall Curve (Decision Tree)", col.main = "darkred")


# 6.3.2 Test Set
# Training the model
tree_model <- rpart(target ~ ., data = train_set, method = "class")
# Making predictions
tree_predictions <- predict(tree_model, test_set, type = "class")
# 7.3.2 Evaluating the model
conf_matrix <-  table(Actual = test_set$target, Predicted = tree_predictions)
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)
# 8.3.2 Confusion matrix
cat("Confusion Matrix:\n", conf_matrix, "\n\n")
# Printing the evaluation metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
# 9.3.2 Visualizing classifier results
# Convert factor predictions to numeric
tree_predictions_numeric <- as.numeric(as.character(tree_predictions))
# ROC Curve for Decision Tree
roc_curve_tree <- roc(test_set$target, tree_predictions_numeric, levels = c(0, 1))
plot(roc_curve_tree, col = "blue", main = "ROC Curve (Decision Tree)", col.main = "darkblue")
# Precision-Recall Curve 
pr_curve_tree <- pr.curve(scores.class0 = tree_predictions, weights.class0 = ifelse(test_set$target == 0, 1, 0), curve = TRUE)
plot(pr_curve_tree, col = "red", main = "Precision-Recall Curve (Decision Tree)", col.main = "darkred")


# 6.4 Random Forest
# 6.4.1 Validation Set
# Training the model
rf_model <- randomForest(target ~ ., data = train_set, ntree = 100)
# Making predictions
rf_validation_predictions <- predict(rf_model, validation_set)
# Converting predictions to binary with a threshold of 0.5
rf_predictions_binary <- ifelse(rf_validation_predictions >= 0.5, 1, 0)
# 7.4.1 Evaluating the model
conf_matrix <-  table(Actual = test_set$target, Predicted = rf_predictions_binary)
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)
# 8.4.1 Confusion matrix
cat("Confusion Matrix:\n", conf_matrix, "\n\n")
# Printing the evaluation metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
# 9.4.1 Visualizing classifier results
# ROC Curve
roc_curve_rf <- roc(test_set$target, rf_predictions_binary, levels = c(0, 1))
plot(roc_curve_rf, col = "blue", main = "ROC Curve (Random Forest)", col.main = "darkblue")
# Precision-Recall Curve 
pr_curve_rf <- pr.curve(scores.class0 = rf_predictions_binary, weights.class0 = ifelse(test_set$target == 0, 1, 0), curve = TRUE)
plot(pr_curve_rf, col = "red", main = "Precision-Recall Curve (Random Forest)", col.main = "darkred")


# 6.4.2 Test Set
# Training the model
rf_model <- randomForest(target ~ ., data = train_set, ntree = 100)
# Making predictions
rf_predictions <- predict(rf_model, test_set)
# Converting predictions to binary with a threshold of 0.5
rf_predictions_binary <- ifelse(rf_predictions >= 0.5, 1, 0)
# 7.4.2 Evaluating the model
conf_matrix <-  table(Actual = test_set$target, Predicted = rf_predictions_binary)
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)
# 8.4.2 Confusion matrix
cat("Confusion Matrix:\n", conf_matrix, "\n\n")
# Printing the evaluation metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
# 9.4.2 Visualizing classifier results
# ROC Curve
roc_curve_rf <- roc(test_set$target, rf_predictions_binary, levels = c(0, 1))
plot(roc_curve_rf, col = "blue", main = "ROC Curve (Random Forest)", col.main = "darkblue")
# Precision-Recall Curve 
pr_curve_rf <- pr.curve(scores.class0 = rf_predictions_binary, weights.class0 = ifelse(test_set$target == 0, 1, 0), curve = TRUE)
plot(pr_curve_rf, col = "red", main = "Precision-Recall Curve (Random Forest)", col.main = "darkred")


# 6.5 XGBoost
# 6.5.1 Validation Set
# Training the model
xgb_model <- xgboost(
  data = as.matrix(X_train),
  label = y_train,
  nrounds = 100, 
  objective = "binary:logistic",
  eval_metric = "logloss",
  verbose = 0
)
# Making predictions
y_pred <- predict(xgb_model, as.matrix(X_validation))
y_pred_binary <- ifelse(y_pred > 0.5, 1, 0)
# 7.5.1 Evaluating the model
conf_matrix <-  table(y_pred_binary, y_validation)
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)
# 8.5.1 Confusion matrix
cat("Confusion Matrix:\n", conf_matrix, "\n\n")
# Printing the evaluation metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
# 9.5.1 Visualizing classifier results
# ROC Curve
roc_curve_boost <- roc(y_validation, y_pred_binary, levels = c(0, 1))
plot(roc_curve_boost, col = "blue", main = "ROC Curve (XGBoost)", col.main = "darkblue")
# Precision-Recall Curve 
pr_curve_boost <- pr.curve(scores.class0 = y_pred_binary, weights.class0 = ifelse(y_validation == 0, 1, 0), curve = TRUE)
plot(pr_curve_boost, col = "red", main = "Precision-Recall Curve (XGBoost)", col.main
