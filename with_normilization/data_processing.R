# Library Imports
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

# Dataframe Assignment
kalp <- read.csv("//path//of//the//kalp.csv")  
head(kalp) 

# 1. Data Analysis and Visualization
summary(kalp) 
# Chest pain type-gender distribution
ggplot(kalp, aes(x = as.factor(cp), fill = as.factor(sex))) +
  geom_bar(position = "dodge") +
  labs(title = "Gender Distribution Based on Chest Pain Type") +
  theme_minimal()
# Resting blood pressure-age distribution
ggplot(kalp, aes(x = age, y = trestbps)) +
  geom_point() +
  labs(title = "Distribution of Resting Blood Pressure by Age") +
  theme_minimal()
# Exercise-induced angina-maximum heart rate
ggplot(kalp, aes(x = as.factor(exang), y = thalach)) +
  geom_boxplot(fill = "lightblue", color = "blue") +
  labs(title = "Maximum Heart Rate by Exercise-Induced Angina Status") +
  theme_minimal()
# Age distribution of patients and non-patients
ggplot(kalp, aes(x = age, fill = as.factor(target))) +
  geom_histogram(position = "stack", bins = 30, alpha = 0.7) +
  labs(title = "Distribution of Age for Patients and Non-Patients") +
  theme_minimal()
# Heart rate-serum cholesterol distribution
ggplot(kalp, aes(x = thalach, y = chol, color = factor(target))) +
  geom_point() +
  labs(title = 'Heart Rate vs Serum Cholesterol Distribution', x = 'Heart Rate', y = 'Serum Cholesterol') +
  scale_color_discrete(labels = c('Not Sick', 'Sick'))
# Exercise-induced angina-ST depression distribution
ggplot(kalp, aes(x = oldpeak, y = as.factor(exang), color = as.factor(target))) +
  geom_point() +
  facet_wrap(~as.factor(target)) +
  labs(
    title = "Distribution of Exercise-Induced Angina and ST Depression",
    x = "ST Depression",
    y = "Exercise-Induced Angina",
    color = "Health Status"
  )
# Fluoroscopy-colored vessel count-health status distribution
ggplot(kalp, aes(x = as.factor(ca), fill = as.factor(target))) +
  geom_bar(position = "fill") +
  labs(
    title = "Distribution of Fluoroscopy-Colored Vessel Count and Health Status",
    x = "Vessel Count",
    y = "Percentage",
    fill = "Health Status"
  )

# 2. Filling Missing Values with Mean - Excluding Target Value
kalp$age[is.na(kalp$age)] <- mean(kalp$age, na.rm = TRUE)
kalp$sex[is.na(kalp$sex)] <- mean(kalp$sex, na.rm = TRUE)
kalp$cp[is.na(kalp$cp)] <- mean(kalp$cp, na.rm = TRUE)
kalp$trestbps[is.na(kalp$trestbps)] <- mean(kalp$trestbps, na.rm = TRUE)
kalp$chol[is.na(kalp$chol)] <- mean(kalp$chol, na.rm = TRUE)
kalp$fbs[is.na(kalp$fbs)] <- mean(kalp$fbs, na.rm = TRUE)
kalp$restecg[is.na(kalp$restecg)] <- mean(kalp$restecg, na.rm = TRUE)
kalp$thalach[is.na(kalp$thalach)] <- mean(kalp$thalach, na.rm = TRUE)
kalp$exang[is.na(kalp$exang)] <- mean(kalp$exang, na.rm = TRUE)
kalp$oldpeak[is.na(kalp$oldpeak)] <- mean(kalp$oldpeak, na.rm = TRUE)
kalp$slope[is.na(kalp$slope)] <- mean(kalp$slope, na.rm = TRUE)
kalp$ca[is.na(kalp$ca)] <- mean(kalp$ca, na.rm = TRUE)
kalp$thal[is.na(kalp$thal)] <- mean(kalp$thal, na.rm = TRUE)

# 3. Normalization 
# Scaling
kalp[-14] <- scale(kalp[-14])  # Normalize all columns except the target column

# 4. Replacing Outliers with IQR (Interquartile Range)
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
# Selecting columns to replace outliers
outlier_columns <- c("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",  "thalach", "exang",  "oldpeak", "slope", "ca", "thal"  )
# Replacing outliers in selected columns
kalp[outlier_columns] <- lapply(kalp[outlier_columns], replace_outliers)

# 5. Adding a New Feature
kalp$new_feature <- kalp$thalach / kalp$age
head(kalp)

# Splitting into Train, Test, and Validation Sets
set.seed(123)
# Creating the Train Set
index_train <- createDataPartition(kalp$target, p = 0.7, list = FALSE)
train_set <- kalp[index_train, ]
# Setting the remaining data as temporary
temp <- kalp[-index_train, ]
# Creating Test and Validation Sets
index_test <- createDataPartition(temp$target, p = 0.5, list = FALSE)
test_set <- temp[index_test, ]
validation_set <- temp[-index_test, ]
