from scipy import stats
import numpy as np

# Example accuracies from 5 folds (in percentage)
fold_accuracies = np.array([78.977,79.907,80.255,78.977,77.816,80.023])

# Define the baseline accuracy (chance level, for example 50%)
baseline = 71.462

# Perform a one-sample t-test against the baseline
t_stat, p_value = stats.ttest_1samp(fold_accuracies, popmean=baseline)

print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.10f}")

mean_accuracy = np.mean(fold_accuracies)
std_accuracy = np.std(fold_accuracies)

print(f"Mean Accuracy: {mean_accuracy:.2f}%")
print(f"Standard Deviation: {std_accuracy:.2f}%")
