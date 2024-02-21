# 10. Finding the model with the highest accuracy
models <- c("kNN", "Random Forest", "SVM", "Decision Tree", "xgBoost")
results_df <- data.frame(Model = character(), Accuracy = numeric(), Precision = numeric(), Recall = numeric(), F1_Score = numeric())
for (model_name in models) {
  if (model_name == "kNN") {
    predictions <- as.numeric(kNN_model)
  } else if (model_name == "Random Forest") {
    predictions <- as.numeric(rf_predictions_binary)
  } else if (model_name == "SVM") {
    predictions <- as.numeric(svm_predictions_binary)
  } else if (model_name == "Decision Tree") {
    predictions <- as.numeric(tree_predictions_numeric)
  } else if (model_name == "xgBoost") {
    predictions <- as.numeric(y_pred_binary)
  }
  
  conf_matrix <- table(predictions, test_set$target)
  accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
  precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
  recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
  f1_score <- 2 * (precision * recall) / (precision + recall)
  
  results_df <- rbind(results_df, data.frame(Model = model_name, Accuracy = accuracy, Precision = precision, Recall = recall, F1_Score = f1_score))
  
}

print(results_df)

ggplot(results_df, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity") +
  labs(title = "Model Comparison", y = "Accuracy") +
  theme_minimal()

# Finding the model with the highest accuracy
best_model <- results_df[which.max(results_df$Accuracy), ]

# Printing information about the model with the highest accuracy
cat("Model with the highest accuracy:\n")
print(best_model)
