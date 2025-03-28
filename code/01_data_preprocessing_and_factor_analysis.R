data <- read.csv(file = 'C:/Users/kimbo/OneDrive - SUNY Upstate Medical University/WMFT_CSM/wmft_all.csv')
#data <- read.csv(file = 'C:/Users/bkjus/OneDrive - SUNY Upstate Medical University/WMFT_CSM/wmft_all.csv')

group <- read.csv(file = 'C:/Users/kimbo/OneDrive - SUNY Upstate Medical University/WMFT_CSM/trial_group.csv')
#group <- read.csv(file = 'C:/Users/bkjus/OneDrive - SUNY Upstate Medical University/WMFT_CSM/trial_group.csv')


chronicity <- read.csv(file = 'C:/Users/kimbo/OneDrive - SUNY Upstate Medical University/WMFT_CSM/chronicity_all.csv')
#chronicity <- read.csv(file = 'C:/Users/bkjus/OneDrive - SUNY Upstate Medical University/WMFT_CSM/chronicity_all.csv')

FM <- read.csv(file = 'C:/Users/kimbo/OneDrive - SUNY Upstate Medical University/WMFT_CSM/FM scores.csv')
#FM <- read.csv(file = 'C:/Users/bkjus/OneDrive - SUNY Upstate Medical University/WMFT_CSM/FM scores.csv')

rate=60/data
rate$group <- group$group
rate$chronicity <- chronicity$chronicity
rate$FM <- FM$FM

# Remove a datapoint which is missing the FM data.
rate<-rate[-264,]

data$group <- group$group
data$chronicity <- chronicity$chronicity
data$FM <- FM$FM
data<-data[-264,]

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

#### Data Pre-processing

## Boxcox transformation for rate data

x <- 1:1:498 # x variable setup

## using the same lambda
b <-boxcox(lm(rate[c(1:15)])) # box cox lambda estimation
lambda <- b$x[which.max(b$y)] # choose the lambda with the maximum log likely hood value
new_data <- (rate[c(1:15)]^lambda - 1)/lambda # BoxCox transformation using the lambda above.
new_data$group<- rate$group
new_data$chronicity<-log(rate$chronicity)
new_data$FM<-rate$FM

result_raw = mvn(data = new_data, scale="TRUE", mvnTest = "energy",
                 univariateTest = "AD", 
                 multivariatePlot = "qq", multivariateOutlierMethod = "adj",
                 showOutliers = TRUE, showNewData = TRUE)

## Remove outliers from MVN
new_data<-result_raw$newData

new_data_normality = mvn(data = new_data, scale="TRUE", mvnTest = "energy",
                                     univariateTest = "AD", 
                                     multivariatePlot = "qq", multivariateOutlierMethod = "adj",
                                     showOutliers = TRUE, showNewData = TRUE)

new_data_normality

new_data<-new_data_normality$newData
bart_spher(new_data) ###### produces Bartletts test of spherecity (you want this to be significant)
KMO(new_data)       ###### Kaiser-Meyer-Olkin measure, you want to be above .7

###EFA

#Exploratory factor analysis in R Dr Paul Christiansen

data2=new_data[c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15")]
number_items <- fa.parallel(data2, fm="ols",fa="fa")
number_items$nfact
##########using Kaisers rule, Eigenvalues>1 represent valid factors


# Final EFA Model
final_model <- fa(scale(data2), nfactors = 2, rotate = "oblimin", fm = "ols")
summary(final_model)
fa.diagram(final_model, main = "EFA with Bootstrapping", digits = 2)
print(final_model, digits = 2, cutoff = 0.5, sort = TRUE)


############### reliability test

factor1_1 <- subset(data2,select=c(V1,V2,V3,V4,V5,V6,V7))

factor2_1 <- subset(data2,select=c(V8,V9,V10,V11,V12,V13,V14,V15))


psych::alpha(factor1_1,check.keys=T)

psych::alpha(factor2_1,check.keys=T)


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
         curvePivot = TRUE, intercepts = FALSE, arrows=TRUE)

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
         edge.width = 1.2)


## validity

mean15<- rowMeans(data2[c(1:15)])
mean4<- rowMeans(data2[c(3, 5, 6, 8)])
mean_data <- data.frame(mean4,mean15)

## 1factor vs. 2factor
anova(fit_boot,fit_boot_2f)

## ICC
ICC(mean_data)

