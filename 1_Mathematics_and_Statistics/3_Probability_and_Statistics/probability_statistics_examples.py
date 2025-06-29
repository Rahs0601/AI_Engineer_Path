# Probability and Statistics Examples using NumPy and SciPy

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 1. Basic Probability
print("\n--- 1. Basic Probability ---")
# Example: Rolling a fair six-sided die
sample_space = {1, 2, 3, 4, 5, 6}

# Event A: Rolling an even number
event_A = {2, 4, 6}
prob_A = len(event_A) / len(sample_space)
print(f"Probability of rolling an even number: {prob_A:.2f}")

# Event B: Rolling a number greater than 3
event_B = {4, 5, 6}
prob_B = len(event_B) / len(sample_space)
print(f"Probability of rolling a number greater than 3: {prob_B:.2f}")

# Event A and B (intersection): Rolling an even number AND greater than 3
event_A_and_B = event_A.intersection(event_B)
prob_A_and_B = len(event_A_and_B) / len(sample_space)
print(f"Probability of rolling an even number AND greater than 3: {prob_A_and_B:.2f}")

# Conditional Probability P(A|B): Probability of A given B
# P(A|B) = P(A and B) / P(B)
prob_A_given_B = prob_A_and_B / prob_B
print(f"Probability of rolling an even number GIVEN it's greater than 3: {prob_A_given_B:.2f}")

# 2. Descriptive Statistics
print("\n--- 2. Descriptive Statistics ---")
data = np.array([10, 12, 12, 13, 15, 16, 18, 20, 22, 22, 25])

print(f"Data: {data}")
print(f"Mean: {np.mean(data):.2f}")
print(f"Median: {np.median(data):.2f}")

# Mode (can be multiple modes)
modes = stats.mode(data)
print(f"Mode: {modes.mode[0]} (count: {modes.count[0]})")

print(f"Variance: {np.var(data):.2f}")
print(f"Standard Deviation: {np.std(data):.2f}")
print(f"Range: {np.max(data) - np.min(data):.2f}")

# 3. Probability Distributions
print("\n--- 3. Probability Distributions ---")

# Bernoulli Distribution (single trial, success/failure)
# Example: Probability of success (p) = 0.7
bernoulli_rv = stats.bernoulli(p=0.7)
print(f"\nBernoulli Distribution (p=0.7):")
print(f"PMF for success (x=1): {bernoulli_rv.pmf(1):.2f}")
print(f"PMF for failure (x=0): {bernoulli_rv.pmf(0):.2f}")

# Binomial Distribution (n trials, k successes)
# Example: 10 coin flips, probability of heads = 0.5. Probability of 7 heads.
n_flips = 10
p_heads = 0.5
binomial_rv = stats.binom(n=n_flips, p=p_heads)
k_heads = 7
print(f"\nBinomial Distribution (n={n_flips}, p={p_heads}):")
print(f"Probability of {k_heads} heads in {n_flips} flips: {binomial_rv.pmf(k_heads):.4f}")

# Normal (Gaussian) Distribution
# Example: Mean = 0, Standard Deviation = 1 (Standard Normal)
mu = 0
sigma = 1
normal_rv = stats.norm(loc=mu, scale=sigma)

print(f"\nNormal Distribution (mu={mu}, sigma={sigma}):")
print(f"PDF at x=0: {normal_rv.pdf(0):.4f}")
print(f"CDF at x=1 (P(X <= 1)): {normal_rv.cdf(1):.4f}")

# Plotting a Normal Distribution (requires matplotlib)
# x = np.linspace(-3, 3, 100)
# plt.plot(x, normal_rv.pdf(x))
# plt.title('Standard Normal Distribution PDF')
# plt.xlabel('x')
# plt.ylabel('Probability Density')
# plt.grid(True)
# plt.show()

# 4. Hypothesis Testing (Independent Samples t-test)
print("\n--- 4. Hypothesis Testing (Independent Samples t-test) ---")
# Example: Do two groups have significantly different means?
# Group A: Test scores of students who used a new study method
group_a_scores = np.array([85, 88, 90, 82, 87, 91, 84, 86, 89, 83])
# Group B: Test scores of students who used the traditional method
group_b_scores = np.array([78, 80, 82, 75, 79, 81, 77, 76, 80, 78])

# Perform independent samples t-test
# Null Hypothesis (H0): The means of the two groups are equal.
# Alternative Hypothesis (H1): The means of the two groups are not equal.

t_statistic, p_value = stats.ttest_ind(group_a_scores, group_b_scores)

alpha = 0.05 # Significance level

print(f"Group A Mean: {np.mean(group_a_scores):.2f}")
print(f"Group B Mean: {np.mean(group_b_scores):.2f}")
print(f"T-statistic: {t_statistic:.3f}")
print(f"P-value: {p_value:.3f}")

if p_value < alpha:
    print(f"Since p-value ({p_value:.3f}) < alpha ({alpha}), we reject the null hypothesis.")
    print("Conclusion: There is a statistically significant difference between the means of the two groups.")
else:
    print(f"Since p-value ({p_value:.3f}) >= alpha ({alpha}), we fail to reject the null hypothesis.")
    print("Conclusion: There is no statistically significant difference between the means of the two groups.")

# 5. Bayes' Theorem Example
print("\n--- 5. Bayes' Theorem ---")
# Scenario: A rare disease affects 1% of the population (P(Disease) = 0.01).
# A test for the disease is 95% accurate (P(Positive|Disease) = 0.95).
# The test has a 10% false positive rate (P(Positive|No Disease) = 0.10).

# We want to find the probability that a person actually has the disease given a positive test result: P(Disease|Positive)

# P(Disease) = Prior probability of having the disease
p_disease = 0.01
# P(No Disease) = 1 - P(Disease)
p_no_disease = 1 - p_disease

# P(Positive|Disease) = Likelihood of positive test given disease (True Positive Rate)
p_pos_given_disease = 0.95

# P(Positive|No Disease) = Likelihood of positive test given no disease (False Positive Rate)
p_pos_given_no_disease = 0.10

# Calculate P(Positive) using the law of total probability:
# P(Positive) = P(Positive|Disease) * P(Disease) + P(Positive|No Disease) * P(No Disease)
p_positive = (p_pos_given_disease * p_disease) + (p_pos_given_no_disease * p_no_disease)

# Apply Bayes' Theorem:
# P(Disease|Positive) = (P(Positive|Disease) * P(Disease)) / P(Positive)
p_disease_given_positive = (p_pos_given_disease * p_disease) / p_positive

print(f"\nGiven:")
print(f"P(Disease): {p_disease}")
print(f"P(Positive|Disease): {p_pos_given_disease}")
print(f"P(Positive|No Disease): {p_pos_given_no_disease}")
print(f"\nCalculated P(Positive): {p_positive:.4f}")
print(f"Probability of having the disease given a positive test (P(Disease|Positive)): {p_disease_given_positive:.4f}")
print("Note: Even with a positive test, the probability of having the rare disease is still relatively low due to its rarity and the false positive rate.")
