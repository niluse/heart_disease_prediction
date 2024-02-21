# Performing operations without normalization

# Assigning the dataframe
heart <- read.csv("//path//of//the//heart.csv")  
head(heart) 

# 1. Data Analysis and Visualization
# We would obtain the same results

# 2. Filling missing values with mean values -excluding the target variable-
heart$age[is.na(heart$age)] <- mean(heart$age, na.rm = TRUE)
heart$sex[is.na(heart$sex)] <- mean(heart$sex, na.rm = TRUE)
heart$cp[is.na(heart$cp)] <- mean(heart$cp, na.rm = TRUE)
heart$trestbps[is.na(heart$trestbps)] <- mean(heart$trestbps, na.rm = TRUE)
heart$chol[is.na(heart$chol)] <- mean(heart$chol, na.rm = TRUE)
heart$fbs[is.na(heart$fbs)] <- mean(heart$fbs, na.rm = TRUE)
heart$restecg[is.na(heart$restecg)] <- mean(heart$restecg, na.rm = TRUE)
heart$thalach[is.na(heart$thalach)] <- mean(heart$thalach, na.rm = TRUE)
heart$exang[is.na(heart$exang)] <- mean(heart$exang, na.rm = TRUE)
heart$oldpeak[is.na(heart$oldpeak)] <- mean(heart$oldpeak, na.rm = TRUE)
heart$slope[is.na(heart$slope)] <- mean(heart$slope, na.rm = TRUE)
heart$ca[is.na(heart$ca)] <- mean(heart$ca, na.rm = TRUE)
heart$thal[is.na(heart$thal)] <- mean(heart$thal, na.rm = TRUE)

# 3. Skipping Normalization
# Skipping this step

# 4. Replacing outliers with IQR (Interquartile Range)
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
# Replacing outliers in the selected columns
heart[outlier_columns] <- lapply(heart[outlier_columns], replace_outliers)

# 5. Adding a new feature
heart$new_feature <- heart$thalach / heart$age
head(heart)

# Splitting into train, test, and validation sets
set.seed(123)
# Creating the Train set
index_train <- createDataPartition(heart$target, p=0.7, list=FALSE)
train_set <- heart[index_train, ]
# Setting the remaining data as temp
temp <- heart[-index_train, ]
# Creating Test and Validation sets
index_test <- createDataPartition(temp$target, p=0.5, list=FALSE)
test_set <- temp[index_test, ]
validation_set <- temp[-index_test, ]
