# 11. Using the best model
# Getting new data
new_data <- read.csv("//path//to//new_data.csv")  
# Making predictions
X_new <- new_data[, !names(new_data) %in% c('target')]  
predictions <- predict(loaded_model, newdata = X_new)
# Displaying predictions
print(predictions)
# Evaluating the model's performance
conf_matrix <-  table(predictions, y_test)
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)
# Confusion matrix
cat("Confusion Matrix:\n", conf_matrix, "\n\n")
# Print the evaluation metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
# Visualizing Classifier Results
# ROC Curve
roc_curve_best <- roc(test_set$target, y_pred_binary, levels = c(0, 1))
plot(roc_curve_best, col = "blue", main = "ROC Curve (XGBoost)", col.main = "darkblue")
# Precision-Recall Curve 
pr_curve_best <- pr.curve(scores.class0 = y_pred_binary, weights.class0 = ifelse(test_set$target == 0, 1, 0), curve = TRUE)
plot(pr_curve_best, col = "red", main = "Precision-Recall Curve (XGBoost)", col.main = "darkred")
