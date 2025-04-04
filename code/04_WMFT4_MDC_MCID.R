# Load necessary packages
library(pwr)
library(dplyr)
library(psych)
library(ggplot2)
library(reshape2)
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



# Enter baseline and follow-up measurements for a group of patients
baseline <- read.csv(file = 'data/WMFT_pre_new.csv', header=FALSE)

post <- read.csv(file = 'data/WMFT_post_new.csv', header=FALSE)

FMA <- read.csv(file = 'data/FM_new.csv', header=FALSE)

FMA_post <- read.csv(file = 'data/FM_post_new.csv', header=FALSE)

## Distribution comparison of WMFT Time scores


melted_data <- melt(baseline)

ggplot(melted_data, aes(x = variable, y = value)) +
  geom_boxplot() +
  labs(title = "Distribution of WMFT Items", x = "WMFT Items", y = "Time Scores [s]") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for better visibility



# Calculate the rate - This was not performed to compare our results with a previous study, which used the raw time.
baseline <- 60/baseline

post <- 60/post


colnames(baseline) <- c("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15")

colnames(post) <- c("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15")

all <- rbind(baseline, post)



melted_data <- melt(baseline)

ggplot(melted_data, aes(x = variable, y = value)) +
  geom_boxplot() +
  labs(title = "Distribution of WMFT Items", x = "WMFT Items", y = "Time Scores [s]") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for better visibility



# Calculate mean difference and standard deviation of differences (WMFT-15)
mean_baseline_15 = rowMeans(baseline)

mean_post_15 = rowMeans(post)

mean_diff_15 <- mean_post_15 - mean_baseline_15

r_15=cor(mean_baseline_15,mean_post_15)

sd_15 <- sd(unlist(all))
sd_pre_15 <- sd(unlist(baseline))
sd_post_15 <- sd(unlist(post))
pooledsd_15 = sqrt((sd_pre_15^2+sd_post_15^2)/2)
mean_of_diff_15 <- mean(mean_diff_15)
sd_of_diff_15 <- sd(mean_diff_15)

# Calculate mean difference and standard deviation of differences (WMFT-4)
mean_baseline_4 = rowMeans(baseline[c(3, 5, 6, 8)])

mean_post_4 = rowMeans(post[c(3, 5, 6, 8)])

mean_diff_4 <- mean_post_4 - mean_baseline_4

r_4=cor(mean_baseline_4,mean_post_4)

sd_4 <- sd(unlist(all[c(3, 5, 6, 8)]))

mean_of_diff_4 <- mean(mean_diff_4)
sd_of_diff_4 <- sd(mean_diff_4)

# Calculate MDC at 90% confidence level - using SEM
MDC_15 <- qt(0.90, df = 396 - 1) * sd_of_diff_15 * sqrt(1-r_15) * sqrt(2)

MDC_15

MDC_4 <- qt(0.90, df = 396 - 1) * sd_of_diff_4 *  sqrt(1-r_4) * sqrt(2)

MDC_4

# Calculate MDC at 95% confidence level - using SEM
MDC_15_95 <- qt(0.95, df = 396 - 1) * sd_of_diff_15 * sqrt(1-r_15) * sqrt(2)

MDC_15_95

MDC_4_95 <- qt(0.95, df = 396 - 1) * sd_of_diff_4 *  sqrt(1-r_4) * sqrt(2)

MDC_4_95

# Calculate MCID based on effect size of 0.2

#MCID_15 <- 0.2 * sd_of_diff_15
#MCID_15

#MCID_4 <-  0.2 * sd_of_diff_4
#MCID_4

# Calculate MCID based on effect size of 0.8

MCID_15 <- 0.8 * sd_of_diff_15
MCID_15

MCID_4 <-  0.8 * sd_of_diff_4
MCID_4

# Print results
cat("MDC_15 at 90% confidence level:", MDC_15, "\n")
cat("MDC_4 at 90% confidence level:", MDC_4, "\n")
cat("MDC_15 at 95% confidence level:", MDC_15_95, "\n")
cat("MDC_4 at 95% confidence level:", MDC_4_95, "\n")
cat("MCID_15 based on effect size of 0.8:", MCID_15, "\n")
cat("MCID_4 based on effect size of 0.8:", MCID_4, "\n")


## Anchor-based MCID using FMA_UE as an anchor

# Define the anchor threshold for a meaningful change - based on the paper by Page, Fulk, and Boyne in Physical Therapy, 2012. We used the CID for the overall UE function in the paper from Table 4.
anchor_threshold <- 5.25

# Calculate FMA pre post difference

FMA_diff <- FMA_post - FMA

# Identify patients who report a meaningful change (anchor responders)
anchor_responders <- FMA_diff >= anchor_threshold

# anchor based MCID for WMFT-6
mean_diff_4 <- data.frame(
  Value = c(mean_diff_4),
  Condition = c(anchor_responders)
)


# Calculate the mean difference between anchor responders and non-responders
mean_by_responders <- tapply(mean_diff_4$Value, mean_diff_4$Condition, mean)

mean_diff <- mean_by_responders["TRUE"] - mean_by_responders["FALSE"]
# Print the anchor-based MCID
print(paste("WMFT-4 MCID_4 (Anchor-Based):", mean_diff))

# Anchor based MCID for WMFT-15
mean_diff_15 <- data.frame(
  Value = c(mean_diff_15),
  Condition = c(anchor_responders)
)


# Calculate the mean difference between anchor responders and non-responders
mean_by_responders_15 <- tapply(mean_diff_15$Value, mean_diff_15$Condition, mean)

mean_diff_15 <- mean_by_responders_15["TRUE"] - mean_by_responders_15["FALSE"]
# Print the anchor-based MCID
print(paste("WMFT-15 MCID_15 (Anchor-Based):", mean_diff_15))


