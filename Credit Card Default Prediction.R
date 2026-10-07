# ============================================================
# CREDIT CARD DEFAULT PREDICTION USING LOGISTIC REGRESSION
# Dataset: UCI Default of Credit Card Clients
# ============================================================

# ============================================================
# 1. INSTALL AND LOAD PACKAGES
# ============================================================

packages <- c(
  "readxl",
  "dplyr",
  "ggplot2",
  "caret",
  "pROC"
)

installed <- rownames(installed.packages())

for (p in packages) {
  if (!(p %in% installed)) {
    install.packages(p)
  }
}

library(readxl)
library(dplyr)
library(ggplot2)
library(caret)
library(pROC)


# ============================================================
# 2. LOAD DATA
# ============================================================

# Download the UCI Excel file manually and place it
# in your working directory.
#
# Dataset:
# "default of credit card clients.xls"
#
# UCI dataset contains 30,000 observations and a binary
# target indicating default in the following month.

data <- read_excel(
  "default of credit card clients.xls",
  skip = 1
)
# Display first observations
head(data)

# Dataset dimensions
dim(data)

# Dataset structure
str(data)


# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================

names(data) <- c(
  "ID",
  "LIMIT_BAL",
  "SEX",
  "EDUCATION",
  "MARRIAGE",
  "AGE",
  "PAY_0",
  "PAY_2",
  "PAY_3",
  "PAY_4",
  "PAY_5",
  "PAY_6",
  "BILL_AMT1",
  "BILL_AMT2",
  "BILL_AMT3",
  "BILL_AMT4",
  "BILL_AMT5",
  "BILL_AMT6",
  "PAY_AMT1",
  "PAY_AMT2",
  "PAY_AMT3",
  "PAY_AMT4",
  "PAY_AMT5",
  "PAY_AMT6",
  "DEFAULT"
)


# ============================================================
# 4. BASIC DATA EXPLORATION
# ============================================================

print(head(data))
print(summary(data))

cat("\nNumber of rows:", nrow(data))
cat("\nNumber of columns:", ncol(data), "\n")

cat("\nMissing values:\n")
print(colSums(is.na(data)))

cat("\nDuplicate rows:", sum(duplicated(data)), "\n")


# ============================================================
# 5. TARGET VARIABLE DISTRIBUTION
# ============================================================

table(data$DEFAULT)

prop.table(table(data$DEFAULT))


# ============================================================
# 6. TARGET VARIABLE VISUALIZATION
# ============================================================

ggplot(data, aes(x = factor(DEFAULT))) +
  geom_bar() +
  labs(
    title = "Credit Card Default Distribution",
    x = "Default Next Month",
    y = "Number of Customers"
  ) +
  theme_minimal()


# ============================================================
# 7. DATA CLEANING
# ============================================================

# Remove duplicate observations
data <- data %>%
  distinct()

# Remove ID because it is only an identifier
data <- data %>%
  select(-ID)


# ============================================================
# 8. HANDLE CATEGORICAL VARIABLES
# ============================================================

data$SEX <- as.factor(data$SEX)

data$EDUCATION <- as.factor(data$EDUCATION)

data$MARRIAGE <- as.factor(data$MARRIAGE)

data$PAY_0 <- as.factor(data$PAY_0)
data$PAY_2 <- as.factor(data$PAY_2)
data$PAY_3 <- as.factor(data$PAY_3)
data$PAY_4 <- as.factor(data$PAY_4)
data$PAY_5 <- as.factor(data$PAY_5)
data$PAY_6 <- as.factor(data$PAY_6)

data$DEFAULT <- as.factor(data$DEFAULT)


# ============================================================
# 9. EXPLORATORY DATA ANALYSIS
# ============================================================

# Credit limit distribution

ggplot(data, aes(x = LIMIT_BAL)) +
  geom_histogram(bins = 40) +
  labs(
    title = "Distribution of Credit Limit",
    x = "Credit Limit",
    y = "Frequency"
  ) +
  theme_minimal()


# Age distribution

ggplot(data, aes(x = AGE)) +
  geom_histogram(bins = 30) +
  labs(
    title = "Age Distribution",
    x = "Age",
    y = "Frequency"
  ) +
  theme_minimal()


# Default rate by education

ggplot(
  data,
  aes(x = EDUCATION, fill = DEFAULT)
) +
  geom_bar(position = "fill") +
  labs(
    title = "Default Rate by Education",
    x = "Education",
    y = "Proportion"
  ) +
  theme_minimal()


# Default rate by sex

ggplot(
  data,
  aes(x = SEX, fill = DEFAULT)
) +
  geom_bar(position = "fill") +
  labs(
    title = "Default Rate by Gender",
    x = "Sex",
    y = "Proportion"
  ) +
  theme_minimal()


# Default rate by marital status

ggplot(
  data,
  aes(x = MARRIAGE, fill = DEFAULT)
) +
  geom_bar(position = "fill") +
  labs(
    title = "Default Rate by Marital Status",
    x = "Marriage Category",
    y = "Proportion"
  ) +
  theme_minimal()


# ============================================================
# 10. TRAIN-TEST SPLIT
# ============================================================

set.seed(123)

# Make sure all levels of PAY variables are represented
# in the training data

repeat {

  train_index <- createDataPartition(
    data$DEFAULT,
    p = 0.80,
    list = FALSE
  )

  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]

  # Check whether every level in test exists in training
  valid_split <- TRUE

  pay_vars <- c(
    "PAY_0", "PAY_2", "PAY_3",
    "PAY_4", "PAY_5", "PAY_6"
  )

  for (v in pay_vars) {

    train_levels <- levels(droplevels(train_data[[v]]))
    test_levels <- levels(droplevels(test_data[[v]]))

    if (!all(test_levels %in% train_levels)) {
      valid_split <- FALSE
      break
    }
  }

  if (valid_split) {
    break
  }
}

cat(
  "\nTraining observations:",
  nrow(train_data)
)

cat(
  "\nTesting observations:",
  nrow(test_data),
  "\n"
)

# ============================================================
# 11. LOGISTIC REGRESSION MODEL
# ============================================================

logistic_model <- glm(
  DEFAULT ~
    LIMIT_BAL +
    SEX +
    EDUCATION +
    MARRIAGE +
    AGE +
    PAY_0 +
    PAY_2 +
    PAY_3 +
    PAY_4 +
    PAY_5 +
    PAY_6 +
    BILL_AMT1 +
    BILL_AMT2 +
    BILL_AMT3 +
    BILL_AMT4 +
    BILL_AMT5 +
    BILL_AMT6 +
    PAY_AMT1 +
    PAY_AMT2 +
    PAY_AMT3 +
    PAY_AMT4 +
    PAY_AMT5 +
    PAY_AMT6,
  data = train_data,
  family = binomial(link = "logit")
)


# ============================================================
# 12. MODEL SUMMARY
# ============================================================

summary(logistic_model)


# ============================================================
# 13. ODDS RATIOS
# ============================================================

odds_ratios <- exp(
  coef(logistic_model)
)

print(odds_ratios)


# ============================================================
# 14. ODDS RATIO CONFIDENCE INTERVALS
# ============================================================

odds_ratio_results <- data.frame(
  Variable = names(coef(logistic_model)),
  Odds_Ratio = exp(coef(logistic_model)),
  Lower_CI = exp(confint.default(logistic_model)[, 1]),
  Upper_CI = exp(confint.default(logistic_model)[, 2])
)

print(odds_ratio_results)


# ============================================================
# 15. PREDICT DEFAULT PROBABILITIES
# ============================================================

predicted_probabilities <- predict(
  logistic_model,
  newdata = test_data,
  type = "response"
)

head(predicted_probabilities)


# ============================================================
# 16. CONVERT PROBABILITIES INTO CLASS PREDICTIONS
# ============================================================

# Threshold = 0.50

predicted_class <- ifelse(
  predicted_probabilities >= 0.50,
  1,
  0
)

predicted_class <- factor(
  predicted_class,
  levels = c(0, 1)
)

actual_class <- factor(
  test_data$DEFAULT,
  levels = c(0, 1)
)


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

confusion_matrix <- confusionMatrix(
  predicted_class,
  actual_class,
  positive = "1"
)

print(confusion_matrix)


# ============================================================
# 18. ACCURACY
# ============================================================

accuracy <- confusion_matrix$overall[
  "Accuracy"
]

cat(
  "\nAccuracy:",
  round(accuracy, 4),
  "\n"
)


# ============================================================
# 19. PRECISION, RECALL AND F1 SCORE
# ============================================================

precision <- confusion_matrix$byClass[
  "Pos Pred Value"
]

recall <- confusion_matrix$byClass[
  "Sensitivity"
]

F1 <- confusion_matrix$byClass[
  "F1"
]

cat(
  "\nPrecision:",
  round(precision, 4)
)

cat(
  "\nRecall:",
  round(recall, 4)
)

cat(
  "\nF1 Score:",
  round(F1, 4),
  "\n"
)


# ============================================================
# 20. ROC CURVE
# ============================================================

roc_curve <- roc(
  test_data$DEFAULT,
  predicted_probabilities
)

print(roc_curve)


# ============================================================
# 21. AUC
# ============================================================

auc_value <- auc(
  roc_curve
)

cat(
  "\nROC-AUC:",
  round(as.numeric(auc_value), 4),
  "\n"
)


# ============================================================
# 22. PLOT ROC CURVE
# ============================================================

plot(
  roc_curve,
  main = "ROC Curve - Logistic Regression"
)

abline(
  a = 0,
  b = 1,
  lty = 2
)


# ============================================================
# 23. PROBABILITY DISTRIBUTION
# ============================================================

prediction_data <- data.frame(
  Actual = test_data$DEFAULT,
  Probability = predicted_probabilities
)

ggplot(
  prediction_data,
  aes(
    x = Probability,
    fill = Actual
  )
) +
  geom_histogram(
    bins = 30,
    alpha = 0.7,
    position = "identity"
  ) +
  labs(
    title = "Predicted Default Probabilities",
    x = "Predicted Probability of Default",
    y = "Frequency"
  ) +
  theme_minimal()


# ============================================================
# 24. FIND IMPORTANT VARIABLES
# ============================================================

# Extract model coefficients
model_coefficients <- coef(logistic_model)

# Extract p-values
model_summary <- summary(logistic_model)$coefficients

# Create p-value vector with same length as coefficients
p_values <- rep(NA, length(model_coefficients))

# Give p-values the same names as coefficients
names(p_values) <- names(model_coefficients)

# Match available p-values to corresponding coefficients
p_values[rownames(model_summary)] <-
  model_summary[, 4]

# Create results table
coefficient_results <- data.frame(
  Variable = names(model_coefficients),
  Coefficient = as.numeric(model_coefficients),
  Odds_Ratio = exp(as.numeric(model_coefficients)),
  P_Value = as.numeric(p_values)
)

# Sort by p-value
coefficient_results <- coefficient_results[
  order(coefficient_results$P_Value,
        na.last = TRUE),
]

print(coefficient_results)

# ============================================================
# 25. SIGNIFICANT VARIABLES
# ============================================================

significant_variables <- coefficient_results[
  coefficient_results$P_Value < 0.05,
]

print(significant_variables)


# ============================================================
# 26. ALTERNATIVE CLASSIFICATION THRESHOLDS
# ============================================================

thresholds <- c(
  0.30,
  0.40,
  0.50,
  0.60,
  0.70
)

threshold_results <- data.frame()

for (threshold in thresholds) {

  pred <- ifelse(
    predicted_probabilities >= threshold,
    1,
    0
  )

  pred <- factor(
    pred,
    levels = c(0, 1)
  )

  cm <- confusionMatrix(
    pred,
    actual_class,
    positive = "1"
  )

  threshold_results <- rbind(
    threshold_results,
    data.frame(
      Threshold = threshold,
      Accuracy = cm$overall["Accuracy"],
      Precision = cm$byClass["Pos Pred Value"],
      Recall = cm$byClass["Sensitivity"],
      F1 = cm$byClass["F1"]
    )
  )
}

print(threshold_results)


# ============================================================
# 27. THRESHOLD COMPARISON
# ============================================================

ggplot(
  threshold_results,
  aes(x = Threshold, y = F1)
) +
  geom_line() +
  geom_point() +
  labs(
    title = "F1 Score Across Classification Thresholds",
    x = "Probability Threshold",
    y = "F1 Score"
  ) +
  theme_minimal()


# ============================================================
# 28. FINAL MODEL PERFORMANCE TABLE
# ============================================================

performance <- data.frame(
  Metric = c(
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
  ),
  Value = c(
    as.numeric(accuracy),
    as.numeric(precision),
    as.numeric(recall),
    as.numeric(F1),
    as.numeric(auc_value)
  )
)

print(performance)


# ============================================================
# 29. SAVE RESULTS
# ============================================================

write.csv(
  performance,
  "logistic_regression_performance.csv",
  row.names = FALSE
)

write.csv(
  odds_ratio_results,
  "odds_ratios.csv",
  row.names = FALSE
)

write.csv(
  coefficient_results,
  "logistic_regression_coefficients.csv",
  row.names = FALSE
)

write.csv(
  threshold_results,
  "threshold_analysis.csv",
  row.names = FALSE
)


# ============================================================
# 30. FINAL OUTPUT
# ============================================================

cat("\n")
cat("============================================================\n")
cat("CREDIT CARD DEFAULT PREDICTION - LOGISTIC REGRESSION\n")
cat("============================================================\n")

cat(
  "\nModel Accuracy:",
  round(as.numeric(accuracy) * 100, 2),
  "%"
)

cat(
  "\nPrecision:",
  round(as.numeric(precision) * 100, 2),
  "%"
)

cat(
  "\nRecall:",
  round(as.numeric(recall) * 100, 2),
  "%"
)

cat(
  "\nF1 Score:",
  round(as.numeric(F1), 4)
)

cat(
  "\nROC-AUC:",
  round(as.numeric(auc_value), 4)
)

cat("\n============================================================\n")