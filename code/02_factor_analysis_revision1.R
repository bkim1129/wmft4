data <- read.csv(file = 'wmft_rate_fm_MVNclean.csv')

rate=60/data

library(caret)
library(boot)
library(MASS)
library(Hmisc)
library(MVN)
library(energy)
library(lavaan)
library(lavaanPlot)
library(semPlot)
library(psych)
library(readxl)
library(REdaS)
library(GPArotation)
library(rio)

new_data <- data
new_data$chronicity <-log(new_data$chronicity)
bart_spher(new_data) ###### produces Bartletts test of spherecity (you want this to be significant)
KMO(new_data)       ###### Kaiser-Meyer-Olkin measure, you want to be above .7

### Confirmatory Factor Analysis
# Define Two-Factor Model
lav.mod2 <- '
F1 =~ V1 + V2 + V3 + V4 + V5 + V6 + V7
F2 =~ V8 + V9 + V10 + V11 + V12 + V13 + V14 + V15
F1 ~~ chronicity
F2 ~~ chronicity
'

# Define One-Factor Model
lav.mod1 <- '
F1 =~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + V11 + V12 + V13 + V14 + V15
F1 ~~ chronicity
'

##  Bootstrapping Analysis
# Define Two-Factor and One-Factor Models
lav.mod2 <- '
F1 =~ V1 + V2 + V3 + V4 + V5 + V6 + V7
F2 =~ V8 + V9 + V10 + V11 + V12 + V13 + V14 + V15
F1 ~~ chronicity
F2 ~~ chronicity
'

lav.mod1 <- '
F1 =~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + V11 + V12 + V13 + V14 + V15
F1 ~~ chronicity
'

# Define Fit Statistic Names and Loading Names
fit_stat_names <- c("CFI", "TLI", "RMSEA", "SRMR", "AIC", "BIC")
variable_names <- c("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15")
loading_names_f1 <- paste("F1", variable_names[1:7], sep = "_")
loading_names_f2 <- paste("F2", variable_names[8:15], sep = "_")
loading_names_1f <- paste("F1", variable_names, sep = "_")

# Bootstrapping Function for Both Models
boot_fn <- function(data, indices, model, is_one_factor = FALSE) {
  sample_data <- data[indices, ]
  fit <- cfa(model, data = scale(sample_data), std.lv = TRUE, estimator = "MLM")
  fit_measures <- fitMeasures(fit, fit_stat_names)
  
  # Extract standardized loadings explicitly for both models
  loadings <- inspect(fit, "std")$lambda
  
  if (is_one_factor) {
    # For one-factor model, extract all 15 loadings
    loadings_vector <- as.vector(loadings[, 1])
  } else {
    # For two-factor model, extract loadings for each factor separately
    f1_loadings <- loadings[1:7, 1]   # Loadings for V1 to V7 on Factor 1
    f2_loadings <- loadings[8:15, 2]  # Loadings for V8 to V15 on Factor 2
    loadings_vector <- c(f1_loadings, f2_loadings)
  }
  
  return(c(fit_measures, loadings_vector))
}

# Perform Bootstrapping for Two-Factor Model
set.seed(123)
bootstrap_2 <- boot(data = new_data, statistic = boot_fn, R = 1000, model = lav.mod2)

# Perform Bootstrapping for One-Factor Model
set.seed(123)
bootstrap_1 <- boot(data = new_data, statistic = function(data, indices) boot_fn(data, indices, lav.mod1, is_one_factor = TRUE), R = 1000)

bootstrap_summary <- function(results_matrix, names_vector) {
  colnames(results_matrix) <- names_vector
  
  mean_values <- colMeans(results_matrix)
  se_values <- apply(results_matrix, 2, sd)
  ci_lower <- apply(results_matrix, 2, quantile, probs = 0.025)
  ci_upper <- apply(results_matrix, 2, quantile, probs = 0.975)
  
  return(data.frame(Name = names_vector,
                    Mean = mean_values,
                    SE = se_values,
                    CI_Lower = ci_lower,
                    CI_Upper = ci_upper))
}

# Extract Fit Statistics and Loadings for Two-Factor Model
n_fit_stats <- length(fit_stat_names)
boot_fit_stats_2 <- bootstrap_2$t[, 1:n_fit_stats]
boot_loadings_f1 <- bootstrap_2$t[, (n_fit_stats + 1):(n_fit_stats + 7)]
boot_loadings_f2 <- bootstrap_2$t[, (n_fit_stats + 8):(n_fit_stats + 15)]

# Extract Fit Statistics and Loadings for One-Factor Model
boot_fit_stats_1 <- bootstrap_1$t[, 1:n_fit_stats]
boot_loadings_1 <- bootstrap_1$t[, (n_fit_stats + 1):(n_fit_stats + length(variable_names))]
loading_names_1f <- paste("F1", variable_names, sep = "_")


fit_stat_names <- c("CFI", "TLI", "RMSEA", "SRMR", "AIC", "BIC")

# Summarize Fit Statistics for Two-Factor Model
cat("\nBootstrapping Summary for Model Fit Statistics (Two-Factor Model):\n")
boot_fit_stats_summary_2 <- bootstrap_summary(boot_fit_stats_2, fit_stat_names)
print(boot_fit_stats_summary_2)

# Summarize Fit Statistics for One-Factor Model
cat("\nBootstrapping Summary for Model Fit Statistics (One-Factor Model):\n")
boot_fit_stats_summary_1 <- bootstrap_summary(boot_fit_stats_1, fit_stat_names)
print(boot_fit_stats_summary_1)


# Summarize Loadings for Factor 1 (Two-Factor Model)
cat("\nBootstrapping Summary for Factor 1 Loadings (Two-Factor Model):\n")
boot_summary_f1 <- bootstrap_summary(boot_loadings_f1, loading_names_f1)
print(boot_summary_f1)

# Summarize Loadings for Factor 2 (Two-Factor Model)
cat("\nBootstrapping Summary for Factor 2 Loadings (Two-Factor Model):\n")
boot_summary_f2 <- bootstrap_summary(boot_loadings_f2, loading_names_f2)
print(boot_summary_f2)

# Summarize Loadings for One-Factor Model
cat("\nBootstrapping Summary for Loadings (One-Factor Model):\n")
boot_summary_1 <- bootstrap_summary(boot_loadings_1, loading_names_1f)
print(boot_summary_1)

## visualization
# Extract the mean loadings from boot_summary_1
mean_loadings <- boot_summary_1$Mean
loading_names <- paste("V", 1:15, sep = "")

# Create a model string dynamically using the bootstrapped mean loadings
model_string <- paste0("F1 =~ ",
                       paste0(round(mean_loadings, 2), "*", loading_names, collapse = " + "),
                       "\nF1 ~~ chronicity")

# Print the model string to verify
cat("CFA Model with Bootstrapped Loadings:\n", model_string, "\n")

# Fit the model using the bootstrapped loadings
fit_boot <- cfa(model_string, data = scale(new_data), std.lv = TRUE, estimator = "MLM")


library(qgraph)
# Create the SEM path diagram using modified graph attributes
semPaths(fit_boot, what = "std", residuals = FALSE, edge.label.cex = 1.2,
         posCol = c("blue", "red"), layout = "circle3", title = FALSE,
         curvePivot = TRUE, intercepts = FALSE, arrows=TRUE, fixedStyle = 1)

## 2factor model visualization
# Load necessary libraries


# Extract the mean loadings from boot_summary_f1 (Factor 1) and boot_summary_f2 (Factor 2)
mean_loadings_f1 <- boot_summary_f1$Mean
mean_loadings_f2 <- boot_summary_f2$Mean

# Define the observed variables for each factor
loading_names_f1 <- paste("V", 1:7, sep = "")  # Variables for Factor 1
loading_names_f2 <- paste("V", 8:15, sep = "") # Variables for Factor 2

# Create the model string using bootstrapped mean loadings
lav_mod <- paste0(
  "F1 =~ ", paste0(round(mean_loadings_f1, 2), "*", loading_names_f1, collapse = " + "), "\n",
  "F2 =~ ", paste0(round(mean_loadings_f2, 2), "*", loading_names_f2, collapse = " + "), "\n",
  "F1 ~~ chronicity\n",
  "F2 ~~ chronicity"
)

# Print the model string for verification
cat("Two-Factor CFA Model with Bootstrapped Loadings:\n", lav_mod, "\n")

# Fit the CFA model using the bootstrapped loadings
fit_boot_2f <- cfa(lav_mod, data = scale(new_data), std.lv = TRUE, estimator = "MLM")

# Create the SEM path diagram with standardized estimates and solid lines
semPaths(fit_boot_2f, what = "std", residuals = FALSE, edge.label.cex = 1.2,
         posCol = c("blue", "red"), layout = "circle3", title = FALSE,
         curvePivot = TRUE, intercepts = FALSE, style = "ram", lty = 1,
         edge.width = 1.2, fixedStyle = 1)

## 1factor vs. 2factor
anova(fit_boot,fit_boot_2f)


### WMFT-4 Factor analysis

WMFT4<- new_data[c(4, 6, 7, 9, 18)]

### Confirmatory Factor Analysis

##  Bootstrapping Analysis
# Define Two-Factor and One-Factor Models
# Define Two-Factor Model
lav.mod2_W4 <- '
F1 =~ V3 + V5 + V6
F2 =~ V8
F1 ~~ chronicity
F2 ~~ chronicity
'

# Define One-Factor Model
lav.mod1_W4 <- '
F1 =~ V3 + V5 + V6 + V8
F1 ~~ chronicity
'

# Bootstrapping Function for Both Models
boot_fn_W4 <- function(data, indices, model, is_one_factor = FALSE) {
  sample_data <- data[indices, ]
  fit <- cfa(model, data = scale(sample_data), std.lv = TRUE, estimator = "MLM")
  fit_measures <- fitMeasures(fit, fit_stat_names)
  
  # Extract standardized loadings explicitly for both models
  loadings <- inspect(fit, "std")$lambda
  
  if (is_one_factor) {
    # For one-factor model, extract all 15 loadings
    loadings_vector <- as.vector(loadings[, 1])
  } else {
    # For two-factor model, extract loadings for each factor separately
    f1_loadings <- loadings[1:3, 1]   # Loadings for V3, 5, 6 on Factor 1
    f2_loadings <- loadings[4, 2]  # Loadings for V8 on Factor 2
    loadings_vector <- c(f1_loadings, f2_loadings)
  }
  
  return(c(fit_measures, loadings_vector))
}

# Define Fit Statistic Names and Loading Names
fit_stat_names_W4 <- c("CFI", "TLI", "RMSEA", "SRMR", "AIC", "BIC")
variable_names_W4 <- c("V3", "V5", "V6", "V8")
loading_names_f1_W4 <- paste("F1", variable_names_W4[1:3], sep = "_")
loading_names_f2_W4 <- paste("F2", variable_names_W4[4], sep = "_")
loading_names_1f_W4 <- paste("F1", variable_names_W4, sep = "_")

# Perform Bootstrapping for Two-Factor Model
set.seed(123)
bootstrap_2_W4 <- boot(data = WMFT4, statistic = boot_fn_W4, R = 1000, model = lav.mod2_W4)

# Perform Bootstrapping for One-Factor Model
set.seed(123)
bootstrap_1_W4 <- boot(data = WMFT4, statistic = function(data, indices) boot_fn_W4(data, indices, lav.mod1_W4, is_one_factor = TRUE), R = 1000)

# Extract Fit Statistics and Loadings for Two-Factor Model
n_fit_stats_W4 <- length(fit_stat_names_W4)
boot_fit_stats_2_W4 <- bootstrap_2_W4$t[, 1:n_fit_stats_W4]
boot_loadings_f1_W4 <- bootstrap_2_W4$t[, (n_fit_stats_W4 + 1):(n_fit_stats_W4 + 3)]
boot_loadings_f2_W4 <- bootstrap_2_W4$t[, (n_fit_stats_W4 + 4)]

# Extract Fit Statistics and Loadings for One-Factor Model
boot_fit_stats_1_W4 <- bootstrap_1_W4$t[, 1:n_fit_stats_W4]
boot_loadings_1_W4 <- bootstrap_1_W4$t[, (n_fit_stats_W4 + 1):(n_fit_stats_W4 + length(variable_names_W4))]
loading_names_1f_W4 <- paste("F1", variable_names_W4, sep = "_")


fit_stat_names_W4 <- c("CFI", "TLI", "RMSEA", "SRMR", "AIC", "BIC")

# Summarize Fit Statistics for Two-Factor Model
cat("\nBootstrapping Summary for Model Fit Statistics (Two-Factor Model):\n")
boot_fit_stats_summary_2_W4 <- bootstrap_summary(boot_fit_stats_2_W4, fit_stat_names_W4)
print(boot_fit_stats_summary_2_W4)

# Summarize Fit Statistics for One-Factor Model
cat("\nBootstrapping Summary for Model Fit Statistics (One-Factor Model):\n")
boot_fit_stats_summary_1_W4 <- bootstrap_summary(boot_fit_stats_1_W4, fit_stat_names_W4)
print(boot_fit_stats_summary_1_W4)


# Summarize Loadings for Factor 1 (Two-Factor Model)
cat("\nBootstrapping Summary for Factor 1 Loadings (Two-Factor Model):\n")
boot_summary_f1_W4 <- bootstrap_summary(boot_loadings_f1_W4, loading_names_f1_W4)
print(boot_summary_f1_W4)

## Summarize Loadings for Factor 2 (Two-Factor Model)
#cat("\nBootstrapping Summary for Factor 2 Loadings (Two-Factor Model):\n")
#boot_summary_f2_W4 <- bootstrap_summary(boot_loadings_f2_W4, loading_names_f2_W4)
#print(boot_summary_f2_W4)

# Summarize Loadings for One-Factor Model
cat("\nBootstrapping Summary for Loadings (One-Factor Model):\n")
boot_summary_1_W4 <- bootstrap_summary(boot_loadings_1_W4, loading_names_1f_W4)
print(boot_summary_1_W4)

## visualization
# Extract the mean loadings from boot_summary_1
mean_loadings_W4 <- boot_summary_1_W4$Mean
loading_names_W4 <- variable_names_W4

# Create a model string dynamically using the bootstrapped mean loadings
model_string_W4 <- paste0("F1 =~ ",
                       paste0(round(mean_loadings_W4, 2), "*", loading_names_W4, collapse = " + "),
                       "\nF1 ~~ chronicity")

# Print the model string to verify
cat("CFA Model with Bootstrapped Loadings:\n", model_string_W4, "\n")

# Fit the model using the bootstrapped loadings
fit_boot_W4 <- cfa(model_string_W4, data = scale(WMFT4), std.lv = TRUE, estimator = "MLM")


library(qgraph)
# Create the SEM path diagram using modified graph attributes
semPaths(fit_boot_W4, what = "std", residuals = FALSE, edge.label.cex = 1.2,
         posCol = c("blue", "red"), layout = "circle3", title = FALSE,
         curvePivot = TRUE, intercepts = FALSE, arrows=TRUE, fixedStyle = 1)

## 2factor model visualization
# Load necessary libraries


# Extract the mean loadings from boot_summary_f1 (Factor 1) and boot_summary_f2 (Factor 2)
mean_loadings_f1_W4 <- boot_summary_f1_W4$Mean
mean_loadings_f2_W4 <- 1

# Define the observed variables for each factor
loading_names_f1_W4 <- variable_names_W4[1:3]  # Variables for Factor 1
loading_names_f2_W4 <- variable_names_W4[4] # Variables for Factor 2

# Create the model string using bootstrapped mean loadings
lav_mod_W4 <- paste0(
  "F1 =~ ", paste0(round(mean_loadings_f1_W4, 2), "*", loading_names_f1_W4, collapse = " + "), "\n",
  "F2 =~ ", paste0(round(mean_loadings_f2_W4, 2), "*", loading_names_f2_W4, collapse = " + "), "\n",
  "F1 ~~ chronicity\n",
  "F2 ~~ chronicity"
)

# Print the model string for verification
cat("Two-Factor CFA Model with Bootstrapped Loadings:\n", lav_mod_W4, "\n")

# Fit the CFA model using the bootstrapped loadings
fit_boot_2f_W4 <- cfa(lav_mod_W4, data = scale(WMFT4), std.lv = TRUE, estimator = "MLM")

# Create the SEM path diagram with standardized estimates and solid lines
semPaths(fit_boot_2f_W4, what = "std", residuals = FALSE, edge.label.cex = 1.2,
         posCol = c("blue", "red"), layout = "circle3", title = FALSE,
         curvePivot = TRUE, intercepts = FALSE, style = "ram", lty = 1,
         edge.width = 1.2, fixedStyle = 1)


## 1factor vs. 2factor
anova(fit_boot_W4,fit_boot_2f_W4)
