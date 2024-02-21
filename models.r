  # Kütüphaneleri kullanıma alinmasi
  library(ggplot2)
  library(ggthemes)
  library(plotly)
  library(dplyr)
  library(psych)
  library(corrplot)
  library(corrr)
  library(tidyverse)
  library(class)
  library(caret)
  library(PRROC)
  library(pROC)
  library(caret)
  library(e1071)
  library(randomForest)
  library(kernlab)
  library(rpart)
  library(randomForest)
  library(xgboost)
  
  # dataframe i atama
  kalp <- read.csv("//home//lily//Desktop//Proje//kalp.csv")  
  head(kalp) 
  
  
  # 1. Veri Analizi ve Görselleştirme
  summary(kalp) 
  # Göğüs ağrısı tipi-cinsiyet dağılımı
  ggplot(kalp, aes(x = as.factor(cp), fill = as.factor(sex))) +
    geom_bar(position = "dodge") +
    labs(title = "Göğüs Ağrısı Tipine Göre Cinsiyet Dağılımı") +
    theme_minimal()
  # Dinlenme kan basıncı-yaş dağılımı
  ggplot(kalp, aes(x = age, y = trestbps)) +
    geom_point() +
    labs(title = "Yaşa Göre Dinlenme Kan Basıncı Dağılımı") +
    theme_minimal()
  # Egzersize bağlı angina-maksimum kalp atış hızı
  ggplot(kalp, aes(x = as.factor(exang), y = thalach)) +
    geom_boxplot(fill = "lightblue", color = "blue") +
    labs(title = "Egzersize Bağlı Angina Durumuna Göre Maksimum Kalp Atış Hızı") +
    theme_minimal()
  # Hasta ve hasta olmayanlar-yaş dağılımı
  ggplot(kalp, aes(x = age, fill = as.factor(target))) +
    geom_histogram(position = "stack", bins = 30, alpha = 0.7) +
    labs(title = "Hasta ve Hasta Olmayanların Yaşa Göre Dağılımı") +
    theme_minimal()
  # kalp atış hızı-serum kolesterol dagilimi
  ggplot(kalp, aes(x=thalach, y=chol, color=factor(target))) +
    geom_point() +
    labs(title='Kalp Atış Hızı ile Serum Kolesterol dagilimi', x='Kalp Atış Hızı', y='Serum Kolesterol') +
    scale_color_discrete(labels=c('Hasta Değil', 'Hasta'))
  # egzersize bağlı angina-ST depresyonu dagilimi
  ggplot(kalp, aes(x = oldpeak, y = as.factor(exang), color = as.factor(target))) +
    geom_point() +
    facet_wrap(~as.factor(target)) +
    labs(
      title = "Egzersize Bağlı Angina ve ST Depresyonu dagilimi",
      x = "ST Depresyonu",
      y = "Egzersize Bağlı Angina",
      color = "Hasta Durumu"
    )
  # floroskopi ile renklendirilen kap sayısı-hasta durumu dagilimi
  ggplot(kalp, aes(x = as.factor(ca), fill = as.factor(target))) +
    geom_bar(position = "fill") +
    labs(
      title = "Floroskopi ile Renklendirilen Kap Sayısı ve Hasta Durumu Dagilimi",
      x = "Kap Sayısı",
      y = "Yüzde",
      fill = "Hasta Durumu"
    )
  
  
  # 2. Eksik verileri ortalama degerler ile doldurma -target degeri haric-
  kalp$age[is.na(kalp$age)]<-mean(kalp$age, na.rm = TRUE)
  kalp$sex[is.na(kalp$sex)]<-mean(kalp$sex, na.rm = TRUE)
  kalp$cp[is.na(kalp$cp)]<-mean(kalp$cp, na.rm = TRUE)
  kalp$trestbps[is.na(kalp$trestbps)]<-mean(kalp$trestbps, na.rm = TRUE)
  kalp$chol[is.na(kalp$chol)]<-mean(kalp$chol, na.rm = TRUE)
  kalp$fbs[is.na(kalp$fbs)]<-mean(kalp$fbs, na.rm = TRUE)
  kalp$restecg[is.na(kalp$restecg)]<-mean(kalp$restecg, na.rm = TRUE)
  kalp$thalach[is.na(kalp$thalach)]<-mean(kalp$thalach, na.rm = TRUE)
  kalp$exang[is.na(kalp$exang)]<-mean(kalp$exang, na.rm = TRUE)
  kalp$oldpeak[is.na(kalp$oldpeak)]<-mean(kalp$oldpeak, na.rm = TRUE)
  kalp$slope[is.na(kalp$slope)]<-mean(kalp$slope, na.rm = TRUE)
  kalp$ca[is.na(kalp$ca)]<-mean(kalp$ca, na.rm = TRUE)
  kalp$thal[is.na(kalp$thal)]<-mean(kalp$thal, na.rm = TRUE)
  
  # 3. Normalizasyon 
  # scaling
  kalp[-14] <- scale(kalp[-14])  # target sütunu hariç tüm sütunların normalize etme

  
  # 4. IQR (Interquartile Range) ile aykırı verileri değiştirme
  replace_outliers <- function(column) {
    Q1 <- quantile(column, 0.25)
    Q3 <- quantile(column, 0.75)
    IQR <- Q3 - Q1
    lower_bound <- Q1 - 1.5 * IQR
    upper_bound <- Q3 + 1.5 * IQR
    column[column < lower_bound] <- max(column[!column < lower_bound])
    column[column > upper_bound] <- max(column[!column > upper_bound])
    return(column)
  }
  # Aykırı değerleri değiştireceğimiz sütunları seçme
  outlier_columns <- c("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",  "thalach", "exang",  "oldpeak", "slope", "ca", "thal"  )
  # Seçilen sütunlardaki aykırı değerleri değiştirme
  kalp[outlier_columns] <- lapply(kalp[outlier_columns], replace_outliers)

  
  # 5. yeni feature ekleme
  kalp$new_feature <- kalp$thalach / kalp$age
  head(kalp)
  
  # test train ve validasyona ayirma
  set.seed(123)
  # Train set oluşturma
  index_train <- createDataPartition(kalp$target, p=0.7, list=FALSE)
  train_set <- kalp[index_train, ]
  # Geriye kalan veriyi temp olarak belirleme
  temp <- kalp[-index_train, ]
  # Test ve validation set oluşturma
  index_test <- createDataPartition(temp$target, p=0.5, list=FALSE)
  test_set <- temp[index_test, ]
  validation_set <- temp[-index_test, ]


# 6. Siniflandirma yapma
# test training ve validasyon setlerinde feature larin ve target in ayrilmasi
X_train <- train_set[, !names(kalp) %in% c('target')]
y_train <- train_set$target
X_test <- test_set[, !names(kalp) %in% c('target')]
y_test <- test_set$target
X_validation <- validation_set[, !names(kalp) %in% c('target')]
y_validation <- validation_set$target


# 6.1 kNN
# 6.1.1. validation_set
# farkli k degerleri ile knn modelini egitme ve degerlendirme
find_best_k <- function(k_values, X_train, y_train, X_validation, y_validation) {
  results <- data.frame(k = numeric(), accuracy = numeric())
  for (k in k_values) {
    model <- knn(train = X_train, test = X_validation, cl = y_train, k = k)
    accuracy <- sum(model == y_validation) / length(y_validation)
    results <- rbind(results, data.frame(k = k, accuracy = accuracy))
  }
  return(results)
}
# test etmek icin k araligi belirleme
k_values <- seq(1, 30, by = 2)
# degiskenlere baslangic deger atama
best_k <- NULL
best_accuracy <- 0
# en iyi k yi bulma
results <- find_best_k(k_values, X_train, y_train, X_validation, y_validation)
# sonuclari yazdirma
print(results)
# En yuksek accuracy olan modeli bulma
best_k_index <- which.max(results$accuracy)
best_accuracy <- results$accuracy[best_k_index]
# en iyi k olan knn 
kNN_model <- knn(train = X_train, test = X_validation, cl = y_train, k = best_k_index)
print(kNN_model)
# 7.1.1 modelin performansinin degerlendirme
conf_matrix <- table(kNN_model, test_set$target)
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)
# 8.1.1 Confusion matrix
cat("Confusion Matrix:\n", conf_matrix, "\n\n")
# Print the evaluation metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
# 9.1.1 Siniflandirici Sonuclarini Gorsellestirme
# ROC Curve
roc_curve <- roc(test_set$target, as.numeric(kNN_model), levels = c(0, 1))
plot(roc_curve, col = "blue", main = "ROC Curve", col.main = "darkblue")
# Precision-Recall Curve
pr_curve <- pr.curve(scores.class0 = as.numeric(kNN_model), weights.class0 = ifelse(test_set$target == 0, 1, 0), curve = TRUE)
plot(pr_curve, col = "red", main = "Precision-Recall Curve", col.main = "darkred")


# 6.1.2. test_set
# farkli k degerleri ile knn modelini egitme ve degerlendirme
find_best_k <- function(k_values, X_train, y_train, X_test, y_test) {
  results <- data.frame(k = numeric(), accuracy = numeric())
  for (k in k_values) {
    model <- knn(train = X_train, test = X_test, cl = y_train, k = k)
    accuracy <- sum(model == y_test) / length(y_test)
    results <- rbind(results, data.frame(k = k, accuracy = accuracy))
  }
  return(results)
}
# test etmek icin k araligi belirleme
k_values <- seq(1, 30, by = 2)
# degiskenlere baslangic deger atama
best_k <- NULL
best_accuracy <- 0
# en iyi k yi bulma
results <- find_best_k(k_values, X_train, y_train, X_test, y_test)
# sonuclari yazdirma
print(results)
# En yuksek accuracy olan modeli bulma
best_k_index <- which.max(results$accuracy)
best_accuracy <- results$accuracy[best_k_index]
# en iyi k olan knn 
kNN_model <- knn(train = X_train, test = X_test, cl = y_train, k = best_k_index)
print(kNN_model)
# 7.1.2 modelin performansinin degerlendirme
conf_matrix <- table(kNN_model, test_set$target)
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)
# 8.1.2 Confusion matrix
cat("Confusion Matrix:\n", conf_matrix, "\n\n")
# Print the evaluation metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
# 9.1.2 Siniflandirici Sonuclarini Gorsellestirme
# ROC Curve
roc_curve <- roc(test_set$target, as.numeric(kNN_model), levels = c(0, 1))
plot(roc_curve, col = "blue", main = "ROC Curve", col.main = "darkblue")
# Precision-Recall Curve
pr_curve <- pr.curve(scores.class0 = as.numeric(kNN_model), weights.class0 = ifelse(test_set$target == 0, 1, 0), curve = TRUE)
plot(pr_curve, col = "red", main = "Precision-Recall Curve", col.main = "darkred")



# 6.2 SVM
# 6.2.1 validation_set
# SVM modelini egitme
svm_model <- svm(target ~ ., data = train_set, kernel = "linear")
# tahminde bulunma
svm_predictions <- predict(svm_model, validation_set)
# threshold u 0.5 alarak tahminleri binary ye cevirme
svm_predictions_binary <- ifelse(svm_predictions >= 0.5, 1, 0)
# 7.2 modelin performansinin degerlendirme
conf_matrix <- table(Actual = test_set$target, Predicted = svm_predictions_binary)
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)
# 8.2 Confusion matrix
cat("Confusion Matrix:\n", conf_matrix, "\n\n")
# Print the evaluation metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
# 9.2 Siniflandirici Sonuclarini Gorsellestirme
# ROC Curve
# ROC Curve for SVM
roc_curve_svm <- roc(test_set$target, svm_predictions, levels = c(0, 1))
plot(roc_curve_svm, col = "blue", main = "ROC Curve (SVM)", col.main = "darkblue")
# Precision-Recall Curve for SVM
pr_curve_svm <- pr.curve(scores.class0 = svm_predictions, weights.class0 = ifelse(test_set$target == 0, 1, 0), curve = TRUE)
plot(pr_curve_svm, col = "red", main = "Precision-Recall Curve (SVM)", col.main = "darkred")

# 6.2.2 test_set
# SVM modelini egitme
svm_model <- svm(target ~ ., data = train_set, kernel = "linear")
# tahminde bulunma
svm_predictions <- predict(svm_model, test_set)
# threshold u 0.5 alarak tahminleri binary ye cevirme
svm_predictions_binary <- ifelse(svm_predictions >= 0.5, 1, 0)
# 7.2.2 modelin performansinin degerlendirme
conf_matrix <- table(Actual = test_set$target, Predicted = svm_predictions_binary)
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)
# 8.2.2 Confusion matrix
cat("Confusion Matrix:\n", conf_matrix, "\n\n")
# Print the evaluation metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
# 9.2.2 Siniflandirici Sonuclarini Gorsellestirme
# ROC Curve
roc_curve_svm <- roc(test_set$target, svm_predictions, levels = c(0, 1))
plot(roc_curve_svm, col = "blue", main = "ROC Curve (SVM)", col.main = "darkblue")
# Precision-Recall Curve for SVM
pr_curve_svm <- pr.curve(scores.class0 = svm_predictions, weights.class0 = ifelse(test_set$target == 0, 1, 0), curve = TRUE)
plot(pr_curve_svm, col = "red", main = "Precision-Recall Curve (SVM)", col.main = "darkred")



# 6.3 Decision Tree
# 6.3.1 validation_set
# modeli egitme
tree_model <- rpart(target ~ ., data = train_set, method = "class")
# tahmin etme
tree_validation_predictions <- predict(tree_model, validation_set, type = "class")
# 7.3.1 modelin performansinin degerlendirilme
conf_matrix <-  table(Actual = test_set$target, Predicted = tree_predictions)
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)
# 8.3.1 Confusion matrix
cat("Confusion Matrix:\n", conf_matrix, "\n\n")
# Print the evaluation metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
# 9.3.1 Siniflandirici Sonuclarini Gorsellestirme
# Convert factor predictions to numeric
tree_predictions_numeric <- as.numeric(as.character(tree_predictions))
# ROC Curve for Decision Tree
roc_curve_tree <- roc(test_set$target, tree_predictions_numeric, levels = c(0, 1))
plot(roc_curve_tree, col = "blue", main = "ROC Curve (Decision Tree)", col.main = "darkblue")
# Precision-Recall Curve 
pr_curve_tree <- pr.curve(scores.class0 = tree_predictions, weights.class0 = ifelse(test_set$target == 0, 1, 0), curve = TRUE)
plot(pr_curve_tree, col = "red", main = "Precision-Recall Curve (Decesion Tree)", col.main = "darkred")



# 6.3.2 test_set
# modeli egitme
tree_model <- rpart(target ~ ., data = train_set, method = "class")
# tahmin etme
tree_predictions <- predict(tree_model, test_set, type = "class")
tree_validation_predictions <- predict(tree_model, validation_set, type = "class")
# 7.3.2 modelin performansinin degerlendirilme
conf_matrix <-  table(Actual = test_set$target, Predicted = tree_predictions)
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)
# 8.3.2 Confusion matrix
cat("Confusion Matrix:\n", conf_matrix, "\n\n")
# Print the evaluation metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
# 9.3.2 Siniflandirici Sonuclarini Gorsellestirme
# Convert factor predictions to numeric
tree_predictions_numeric <- as.numeric(as.character(tree_predictions))
# ROC Curve for Decision Tree
roc_curve_tree <- roc(test_set$target, tree_predictions_numeric, levels = c(0, 1))
plot(roc_curve_tree, col = "blue", main = "ROC Curve (Decision Tree)", col.main = "darkblue")
# Precision-Recall Curve 
pr_curve_tree <- pr.curve(scores.class0 = tree_predictions, weights.class0 = ifelse(test_set$target == 0, 1, 0), curve = TRUE)
plot(pr_curve_tree, col = "red", main = "Precision-Recall Curve (Decesion Tree)", col.main = "darkred")




# 6.4 random forest
# 6.4.1 validation_set
# modeli egitme
rf_model <- randomForest(target ~ ., data = train_set, ntree = 100)
# tahmin etme
rf_validation_predictions <- predict(rf_model, validation_set)
# threshold a 0.5 vererek tahminleri binary yapma
rf_predictions_binary <- ifelse(rf_predictions >= 0.5, 1, 0)
# 7.4.1 modelin performansinin degerlendirme
conf_matrix <-  table(Actual = test_set$target, Predicted = rf_predictions_binary)
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)
# 8.4.1 Confusion matrix
cat("Confusion Matrix:\n", conf_matrix, "\n\n")
# Print the evaluation metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
# 9.4.1 Siniflandirici Sonuclarini Gorsellestirme
# ROC Curve
roc_curve_rf <- roc(test_set$target, rf_predictions_binary, levels = c(0, 1))
plot(roc_curve_svm, col = "blue", main = "ROC Curve (Random Forest)", col.main = "darkblue")
# Precision-Recall Curve 
pr_curve_rf <- pr.curve(scores.class0 = rf_predictions_binary, weights.class0 = ifelse(test_set$target == 0, 1, 0), curve = TRUE)
plot(pr_curve_svm, col = "red", main = "Precision-Recall Curve (Random Forest)", col.main = "darkred")




# 6.4.2 test_set
# modeli egitme
rf_model <- randomForest(target ~ ., data = train_set, ntree = 100)
# tahmin etme
rf_predictions <- predict(rf_model, test_set)
rf_validation_predictions <- predict(rf_model, validation_set)
# threshold a 0.5 vererek tahminleri binary yapma
rf_predictions_binary <- ifelse(rf_predictions >= 0.5, 1, 0)
# 7.4.2 modelin performansinin degerlendirme
conf_matrix <-  table(Actual = test_set$target, Predicted = rf_predictions_binary)
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)
# 8.4.2 Confusion matrix
cat("Confusion Matrix:\n", conf_matrix, "\n\n")
# Print the evaluation metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
# 9.4.2 Siniflandirici Sonuclarini Gorsellestirme
# ROC Curve
roc_curve_rf <- roc(test_set$target, rf_predictions_binary, levels = c(0, 1))
plot(roc_curve_rf, col = "blue", main = "ROC Curve (Random Forest)", col.main = "darkblue")
# Precision-Recall Curve 
pr_curve_rf <- pr.curve(scores.class0 = rf_predictions_binary, weights.class0 = ifelse(test_set$target == 0, 1, 0), curve = TRUE)
plot(pr_curve_rf, col = "red", main = "Precision-Recall Curve (Random Forest)", col.main = "darkred")


# 6.5 xgboost
# 6.5.1 validation_set
# modeli egitme
xgb_model <- xgboost(
  data = as.matrix(X_train),
  label = y_train,
  nrounds = 100, 
  objective = "binary:logistic",
  eval_metric = "logloss",
  verbose = 0
)
# tahminde bulunma
y_pred <- predict(xgb_model, as.matrix(X_validation))
y_pred_binary <- ifelse(y_pred > 0.5, 1, 0)
# 7.5.1 modelin performansinin degerlendirilmesi
conf_matrix <-  table(y_pred_binary, y_test)
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)
# 8.5.1 Confusion matrix
cat("Confusion Matrix:\n", conf_matrix, "\n\n")
# Print the evaluation metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
# 9.5.1 Siniflandirici Sonuclarini Gorsellestirme
# ROC Curve
roc_curve_boost <- roc(test_set$target, y_pred_binary, levels = c(0, 1))
plot(roc_curve_boost, col = "blue", main = "ROC Curve (XGBoost)", col.main = "darkblue")
# Precision-Recall Curve 
pr_curve_boost <- pr.curve(scores.class0 = y_pred_binary, weights.class0 = ifelse(test_set$target == 0, 1, 0), curve = TRUE)
plot(pr_curve_boost, col = "red", main = "Precision-Recall Curve (XGBoost)", col.main = "darkred")



# 6.5.2 test_set
# modeli egitme
xgb_model <- xgboost(
  data = as.matrix(X_train),
  label = y_train,
  nrounds = 100, # You can adjust the number of rounds as needed
  objective = "binary:logistic",
  eval_metric = "logloss",
  verbose = 0
)
# tahminde bulunma
y_pred <- predict(xgb_model, as.matrix(X_test))
y_pred_binary <- ifelse(y_pred > 0.5, 1, 0)
# 7.5.2 modelin performansinin degerlendirilmesi
conf_matrix <-  table(y_pred_binary, y_test)
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)
# 8.5.2 Confusion matrix
cat("Confusion Matrix:\n", conf_matrix, "\n\n")
# Print the evaluation metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
# 9.5.2 Siniflandirici Sonuclarini Gorsellestirme
# ROC Curve
roc_curve_boost <- roc(test_set$target, y_pred_binary, levels = c(0, 1))
plot(roc_curve_boost, col = "blue", main = "ROC Curve (XGBoost)", col.main = "darkblue")
# Precision-Recall Curve 
pr_curve_boost <- pr.curve(scores.class0 = y_pred_binary, weights.class0 = ifelse(test_set$target == 0, 1, 0), curve = TRUE)
plot(pr_curve_boost, col = "red", main = "Precision-Recall Curve (XGBoost)", col.main = "darkred")



# 10. En yüksek doğruluk değerine sahip modeli bulma
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

# En yüksek doğruluk değerine sahip modeli bulma
best_model <- results_df[which.max(results_df$Accuracy), ]

# En yüksek doğruluk değerine sahip modelin bilgilerini yazdırma
cat("En yüksek doğruluk değerine sahip model:\n")
print(best_model)



# 11. en iyi modeli kullanma
# yeni datalarin alinmasi
new_data <- read.csv("//path//to//new_data.csv")  
# tahminde bulunma
X_new <- new_data[, !names(new_data) %in% c('target')]  
predictions <- predict(loaded_model, newdata = X_new)
# tahminlerin goruntulenmesi
print(predictions)
# modelin performansinin degerlendirilmesi
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
# Siniflandirici Sonuclarini Gorsellestirme
# ROC Curve
roc_curve_best <- roc(test_set$target, y_pred_binary, levels = c(0, 1))
plot(roc_curve_best, col = "blue", main = "ROC Curve (XGBoost)", col.main = "darkblue")
# Precision-Recall Curve 
pr_curve_best <- pr.curve(scores.class0 = y_pred_binary, weights.class0 = ifelse(test_set$target == 0, 1, 0), curve = TRUE)
plot(pr_curve_best, col = "red", main = "Precision-Recall Curve (XGBoost)", col.main = "darkred")




################################################################################
# Normalizasyon yapmadan islemleri gerceklestirme

# dataframe i atama
kalp <- read.csv("//home//lily//Desktop//Proje//kalp.csv")  
head(kalp) 


# 1. Veri Analizi ve Görselleştirme
# ayni sonuclari aliriz

# 2. Eksik verileri ortalama degerler ile doldurma -target degeri haric-
kalp$age[is.na(kalp$age)]<-mean(kalp$age, na.rm = TRUE)
kalp$sex[is.na(kalp$sex)]<-mean(kalp$sex, na.rm = TRUE)
kalp$cp[is.na(kalp$cp)]<-mean(kalp$cp, na.rm = TRUE)
kalp$trestbps[is.na(kalp$trestbps)]<-mean(kalp$trestbps, na.rm = TRUE)
kalp$chol[is.na(kalp$chol)]<-mean(kalp$chol, na.rm = TRUE)
kalp$fbs[is.na(kalp$fbs)]<-mean(kalp$fbs, na.rm = TRUE)
kalp$restecg[is.na(kalp$restecg)]<-mean(kalp$restecg, na.rm = TRUE)
kalp$thalach[is.na(kalp$thalach)]<-mean(kalp$thalach, na.rm = TRUE)
kalp$exang[is.na(kalp$exang)]<-mean(kalp$exang, na.rm = TRUE)
kalp$oldpeak[is.na(kalp$oldpeak)]<-mean(kalp$oldpeak, na.rm = TRUE)
kalp$slope[is.na(kalp$slope)]<-mean(kalp$slope, na.rm = TRUE)
kalp$ca[is.na(kalp$ca)]<-mean(kalp$ca, na.rm = TRUE)
kalp$thal[is.na(kalp$thal)]<-mean(kalp$thal, na.rm = TRUE)

# 3. Normalizasyon 
# bu adimi atliyoruz

# 4. IQR (Interquartile Range) ile aykırı verileri değiştirme
replace_outliers <- function(column) {
  Q1 <- quantile(column, 0.25)
  Q3 <- quantile(column, 0.75)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  column[column < lower_bound] <- max(column[!column < lower_bound])
  column[column > upper_bound] <- max(column[!column > upper_bound])
  return(column)
}
# Aykırı değerleri değiştireceğimiz sütunları seçme
outlier_columns <- c("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",  "thalach", "exang",  "oldpeak", "slope", "ca", "thal"  )
# Seçilen sütunlardaki aykırı değerleri değiştirme
kalp[outlier_columns] <- lapply(kalp[outlier_columns], replace_outliers)


# 5. yeni feature ekleme
kalp$new_feature <- kalp$thalach / kalp$age
head(kalp)

# test train ve validasyona ayirma
set.seed(123)
# Train set oluşturma
index_train <- createDataPartition(kalp$target, p=0.7, list=FALSE)
train_set <- kalp[index_train, ]
# Geriye kalan veriyi temp olarak belirleme
temp <- kalp[-index_train, ]
# Test ve validation set oluşturma
index_test <- createDataPartition(temp$target, p=0.5, list=FALSE)
test_set <- temp[index_test, ]
validation_set <- temp[-index_test, ]


# 6. Siniflandirma yapma
# test training ve validasyon setlerinde feature larin ve target in ayrilmasi
X_train <- train_set[, !names(kalp) %in% c('target')]
y_train <- train_set$target
X_test <- test_set[, !names(kalp) %in% c('target')]
y_test <- test_set$target
X_validation <- validation_set[, !names(kalp) %in% c('target')]
y_validation <- validation_set$target


# 6.1 kNN
# 6.1.1. validation_set
# farkli k degerleri ile knn modelini egitme ve degerlendirme
find_best_k <- function(k_values, X_train, y_train, X_validation, y_validation) {
  results <- data.frame(k = numeric(), accuracy = numeric())
  for (k in k_values) {
    model <- knn(train = X_train, test = X_validation, cl = y_train, k = k)
    accuracy <- sum(model == y_validation) / length(y_validation)
    results <- rbind(results, data.frame(k = k, accuracy = accuracy))
  }
  return(results)
}
# test etmek icin k araligi belirleme
k_values <- seq(1, 30, by = 2)
# degiskenlere baslangic deger atama
best_k <- NULL
best_accuracy <- 0
# en iyi k yi bulma
results <- find_best_k(k_values, X_train, y_train, X_validation, y_validation)
# sonuclari yazdirma
print(results)
# En yuksek accuracy olan modeli bulma
best_k_index <- which.max(results$accuracy)
best_accuracy <- results$accuracy[best_k_index]
# en iyi k olan knn 
kNN_model <- knn(train = X_train, test = X_validation, cl = y_train, k = best_k_index)
print(kNN_model)
# 7.1.1 modelin performansinin degerlendirme
conf_matrix <- table(kNN_model, test_set$target)
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)
# 8.1.1 Confusion matrix
cat("Confusion Matrix:\n", conf_matrix, "\n\n")
# Print the evaluation metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
# 9.1.1 Siniflandirici Sonuclarini Gorsellestirme
# ROC Curve
roc_curve <- roc(test_set$target, as.numeric(kNN_model), levels = c(0, 1))
plot(roc_curve, col = "blue", main = "ROC Curve", col.main = "darkblue")
# Precision-Recall Curve
pr_curve <- pr.curve(scores.class0 = as.numeric(kNN_model), weights.class0 = ifelse(test_set$target == 0, 1, 0), curve = TRUE)
plot(pr_curve, col = "red", main = "Precision-Recall Curve", col.main = "darkred")


# 6.1.2. test_set
# farkli k degerleri ile knn modelini egitme ve degerlendirme
find_best_k <- function(k_values, X_train, y_train, X_test, y_test) {
  results <- data.frame(k = numeric(), accuracy = numeric())
  for (k in k_values) {
    model <- knn(train = X_train, test = X_test, cl = y_train, k = k)
    accuracy <- sum(model == y_test) / length(y_test)
    results <- rbind(results, data.frame(k = k, accuracy = accuracy))
  }
  return(results)
}
# test etmek icin k araligi belirleme
k_values <- seq(1, 30, by = 2)
# degiskenlere baslangic deger atama
best_k <- NULL
best_accuracy <- 0
# en iyi k yi bulma
results <- find_best_k(k_values, X_train, y_train, X_test, y_test)
# sonuclari yazdirma
print(results)
# En yuksek accuracy olan modeli bulma
best_k_index <- which.max(results$accuracy)
best_accuracy <- results$accuracy[best_k_index]
# en iyi k olan knn 
kNN_model <- knn(train = X_train, test = X_test, cl = y_train, k = best_k_index)
print(kNN_model)
# 7.1.2 modelin performansinin degerlendirme
conf_matrix <- table(kNN_model, test_set$target)
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)
# 8.1.2 Confusion matrix
cat("Confusion Matrix:\n", conf_matrix, "\n\n")
# Print the evaluation metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
# 9.1.2 Siniflandirici Sonuclarini Gorsellestirme
# ROC Curve
roc_curve <- roc(test_set$target, as.numeric(kNN_model), levels = c(0, 1))
plot(roc_curve, col = "blue", main = "ROC Curve", col.main = "darkblue")
# Precision-Recall Curve
pr_curve <- pr.curve(scores.class0 = as.numeric(kNN_model), weights.class0 = ifelse(test_set$target == 0, 1, 0), curve = TRUE)
plot(pr_curve, col = "red", main = "Precision-Recall Curve", col.main = "darkred")



# 6.2 SVM
# 6.2.1 validation_set
# SVM modelini egitme
svm_model <- svm(target ~ ., data = train_set, kernel = "linear")
# tahminde bulunma
svm_predictions <- predict(svm_model, validation_set)
# threshold u 0.5 alarak tahminleri binary ye cevirme
svm_predictions_binary <- ifelse(svm_predictions >= 0.5, 1, 0)
# 7.2 modelin performansinin degerlendirme
conf_matrix <- table(Actual = test_set$target, Predicted = svm_predictions_binary)
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)
# 8.2 Confusion matrix
cat("Confusion Matrix:\n", conf_matrix, "\n\n")
# Print the evaluation metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
# 9.2 Siniflandirici Sonuclarini Gorsellestirme
# ROC Curve
# ROC Curve for SVM
roc_curve_svm <- roc(test_set$target, svm_predictions, levels = c(0, 1))
plot(roc_curve_svm, col = "blue", main = "ROC Curve (SVM)", col.main = "darkblue")
# Precision-Recall Curve for SVM
pr_curve_svm <- pr.curve(scores.class0 = svm_predictions, weights.class0 = ifelse(test_set$target == 0, 1, 0), curve = TRUE)
plot(pr_curve_svm, col = "red", main = "Precision-Recall Curve (SVM)", col.main = "darkred")

# 6.2.2 test_set
# SVM modelini egitme
svm_model <- svm(target ~ ., data = train_set, kernel = "linear")
# tahminde bulunma
svm_predictions <- predict(svm_model, test_set)
# threshold u 0.5 alarak tahminleri binary ye cevirme
svm_predictions_binary <- ifelse(svm_predictions >= 0.5, 1, 0)
# 7.2.2 modelin performansinin degerlendirme
conf_matrix <- table(Actual = test_set$target, Predicted = svm_predictions_binary)
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)
# 8.2.2 Confusion matrix
cat("Confusion Matrix:\n", conf_matrix, "\n\n")
# Print the evaluation metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
# 9.2.2 Siniflandirici Sonuclarini Gorsellestirme
# ROC Curve
roc_curve_svm <- roc(test_set$target, svm_predictions, levels = c(0, 1))
plot(roc_curve_svm, col = "blue", main = "ROC Curve (SVM)", col.main = "darkblue")
# Precision-Recall Curve for SVM
pr_curve_svm <- pr.curve(scores.class0 = svm_predictions, weights.class0 = ifelse(test_set$target == 0, 1, 0), curve = TRUE)
plot(pr_curve_svm, col = "red", main = "Precision-Recall Curve (SVM)", col.main = "darkred")



# 6.3 Decision Tree
# 6.3.1 validation_set
# modeli egitme
tree_model <- rpart(target ~ ., data = train_set, method = "class")
# tahmin etme
tree_validation_predictions <- predict(tree_model, validation_set, type = "class")
# 7.3.1 modelin performansinin degerlendirilme
conf_matrix <-  table(Actual = test_set$target, Predicted = tree_predictions)
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)
# 8.3.1 Confusion matrix
cat("Confusion Matrix:\n", conf_matrix, "\n\n")
# Print the evaluation metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
# 9.3.1 Siniflandirici Sonuclarini Gorsellestirme
# Convert factor predictions to numeric
tree_predictions_numeric <- as.numeric(as.character(tree_predictions))
# ROC Curve for Decision Tree
roc_curve_tree <- roc(test_set$target, tree_predictions_numeric, levels = c(0, 1))
plot(roc_curve_tree, col = "blue", main = "ROC Curve (Decision Tree)", col.main = "darkblue")
# Precision-Recall Curve 
pr_curve_tree <- pr.curve(scores.class0 = tree_predictions, weights.class0 = ifelse(test_set$target == 0, 1, 0), curve = TRUE)
plot(pr_curve_tree, col = "red", main = "Precision-Recall Curve (Decesion Tree)", col.main = "darkred")



# 6.3.2 test_set
# modeli egitme
tree_model <- rpart(target ~ ., data = train_set, method = "class")
# tahmin etme
tree_predictions <- predict(tree_model, test_set, type = "class")
tree_validation_predictions <- predict(tree_model, validation_set, type = "class")
# 7.3.2 modelin performansinin degerlendirilme
conf_matrix <-  table(Actual = test_set$target, Predicted = tree_predictions)
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)
# 8.3.2 Confusion matrix
cat("Confusion Matrix:\n", conf_matrix, "\n\n")
# Print the evaluation metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
# 9.3.2 Siniflandirici Sonuclarini Gorsellestirme
# Convert factor predictions to numeric
tree_predictions_numeric <- as.numeric(as.character(tree_predictions))
# ROC Curve for Decision Tree
roc_curve_tree <- roc(test_set$target, tree_predictions_numeric, levels = c(0, 1))
plot(roc_curve_tree, col = "blue", main = "ROC Curve (Decision Tree)", col.main = "darkblue")
# Precision-Recall Curve 
pr_curve_tree <- pr.curve(scores.class0 = tree_predictions, weights.class0 = ifelse(test_set$target == 0, 1, 0), curve = TRUE)
plot(pr_curve_tree, col = "red", main = "Precision-Recall Curve (Decesion Tree)", col.main = "darkred")




# 6.4 random forest
# 6.4.1 validation_set
# modeli egitme
rf_model <- randomForest(target ~ ., data = train_set, ntree = 100)
# tahmin etme
rf_validation_predictions <- predict(rf_model, validation_set)
# threshold a 0.5 vererek tahminleri binary yapma
rf_predictions_binary <- ifelse(rf_predictions >= 0.5, 1, 0)
# 7.4.1 modelin performansinin degerlendirme
conf_matrix <-  table(Actual = test_set$target, Predicted = rf_predictions_binary)
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)
# 8.4.1 Confusion matrix
cat("Confusion Matrix:\n", conf_matrix, "\n\n")
# Print the evaluation metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
# 9.4.1 Siniflandirici Sonuclarini Gorsellestirme
# ROC Curve
roc_curve_rf <- roc(test_set$target, rf_predictions_binary, levels = c(0, 1))
plot(roc_curve_svm, col = "blue", main = "ROC Curve (Random Forest)", col.main = "darkblue")
# Precision-Recall Curve 
pr_curve_rf <- pr.curve(scores.class0 = rf_predictions_binary, weights.class0 = ifelse(test_set$target == 0, 1, 0), curve = TRUE)
plot(pr_curve_svm, col = "red", main = "Precision-Recall Curve (Random Forest)", col.main = "darkred")




# 6.4.2 test_set
# modeli egitme
rf_model <- randomForest(target ~ ., data = train_set, ntree = 100)
# tahmin etme
rf_predictions <- predict(rf_model, test_set)
rf_validation_predictions <- predict(rf_model, validation_set)
# threshold a 0.5 vererek tahminleri binary yapma
rf_predictions_binary <- ifelse(rf_predictions >= 0.5, 1, 0)
# 7.4.2 modelin performansinin degerlendirme
conf_matrix <-  table(Actual = test_set$target, Predicted = rf_predictions_binary)
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)
# 8.4.2 Confusion matrix
cat("Confusion Matrix:\n", conf_matrix, "\n\n")
# Print the evaluation metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
# 9.4.2 Siniflandirici Sonuclarini Gorsellestirme
# ROC Curve
roc_curve_rf <- roc(test_set$target, rf_predictions_binary, levels = c(0, 1))
plot(roc_curve_rf, col = "blue", main = "ROC Curve (Random Forest)", col.main = "darkblue")
# Precision-Recall Curve 
pr_curve_rf <- pr.curve(scores.class0 = rf_predictions_binary, weights.class0 = ifelse(test_set$target == 0, 1, 0), curve = TRUE)
plot(pr_curve_rf, col = "red", main = "Precision-Recall Curve (Random Forest)", col.main = "darkred")


# 6.5 xgboost
# 6.5.1 validation_set
# modeli egitme
xgb_model <- xgboost(
  data = as.matrix(X_train),
  label = y_train,
  nrounds = 100, 
  objective = "binary:logistic",
  eval_metric = "logloss",
  verbose = 0
)
# tahminde bulunma
y_pred <- predict(xgb_model, as.matrix(X_validation))
y_pred_binary <- ifelse(y_pred > 0.5, 1, 0)
# 7.5.1 modelin performansinin degerlendirilmesi
conf_matrix <-  table(y_pred_binary, y_test)
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)
# 8.5.1 Confusion matrix
cat("Confusion Matrix:\n", conf_matrix, "\n\n")
# Print the evaluation metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
# 9.5.1 Siniflandirici Sonuclarini Gorsellestirme
# ROC Curve
roc_curve_boost <- roc(test_set$target, y_pred_binary, levels = c(0, 1))
plot(roc_curve_boost, col = "blue", main = "ROC Curve (XGBoost)", col.main = "darkblue")
# Precision-Recall Curve 
pr_curve_boost <- pr.curve(scores.class0 = y_pred_binary, weights.class0 = ifelse(test_set$target == 0, 1, 0), curve = TRUE)
plot(pr_curve_boost, col = "red", main = "Precision-Recall Curve (XGBoost)", col.main = "darkred")



# 6.5.2 test_set
# modeli egitme
xgb_model <- xgboost(
  data = as.matrix(X_train),
  label = y_train,
  nrounds = 100, # You can adjust the number of rounds as needed
  objective = "binary:logistic",
  eval_metric = "logloss",
  verbose = 0
)
# tahminde bulunma
y_pred <- predict(xgb_model, as.matrix(X_test))
y_pred_binary <- ifelse(y_pred > 0.5, 1, 0)
# 7.5.2 modelin performansinin degerlendirilmesi
conf_matrix <-  table(y_pred_binary, y_test)
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)
# 8.5.2 Confusion matrix
cat("Confusion Matrix:\n", conf_matrix, "\n\n")
# Print the evaluation metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
# 9.5.2 Siniflandirici Sonuclarini Gorsellestirme
# ROC Curve
roc_curve_boost <- roc(test_set$target, y_pred_binary, levels = c(0, 1))
plot(roc_curve_boost, col = "blue", main = "ROC Curve (XGBoost)", col.main = "darkblue")
# Precision-Recall Curve 
pr_curve_boost <- pr.curve(scores.class0 = y_pred_binary, weights.class0 = ifelse(test_set$target == 0, 1, 0), curve = TRUE)
plot(pr_curve_boost, col = "red", main = "Precision-Recall Curve (XGBoost)", col.main = "darkred")



# 10. En yüksek doğruluk değerine sahip modeli bulma
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
# En yüksek doğruluk değerine sahip modeli bulma
best_model <- results_df[which.max(results_df$Accuracy), ]
# En yüksek doğruluk değerine sahip modelin bilgilerini yazdırma
cat("En yüksek doğruluk değerine sahip model:\n")
print(best_model)

# normalizasyon yapilmadiginda en iyi sonuc decesion tree en iyi accuracy yi verirken normalizasyon ile knn modeli en iyi accuracy sonucunu almaktadir



# 11. en iyi modeli kullanma
# yeni datalarin alinmasi
new_data <- read.csv("//path//to//new_data.csv")  
# tahminde bulunma
X_new <- new_data[, !names(new_data) %in% c('target')]  
predictions <- predict(loaded_model, newdata = X_new)
# tahminlerin goruntulenmesi
print(predictions)
# modelin performansinin degerlendirilmesi
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
# Siniflandirici Sonuclarini Gorsellestirme
# ROC Curve
roc_curve_best <- roc(test_set$target, y_pred_binary, levels = c(0, 1))
plot(roc_curve_best, col = "blue", main = "ROC Curve (XGBoost)", col.main = "darkblue")
# Precision-Recall Curve 
pr_curve_best <- pr.curve(scores.class0 = y_pred_binary, weights.class0 = ifelse(test_set$target == 0, 1, 0), curve = TRUE)
plot(pr_curve_best, col = "red", main = "Precision-Recall Curve (XGBoost)", col.main = "darkred")

