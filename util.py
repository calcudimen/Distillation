import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def generate_gaussian_variables(n, mean=0, std_dev=1):
    """
    Generate Dn Gaussian random variables.

    Parameters:
    - n: Number of random variables to generate.
    - mean: Mean of the Gaussian distribution (default is 0).
    - std_dev: Standard deviation of the Gaussian distribution (default is 1).

    Returns:
    A NumPy array containing Dn Gaussian random variables.
    """
    return np.random.normal(mean, std_dev, n)

def generate_data(N, Dn, true_feature):
    """
    Generate synthetic data for logistic regression.

    Args:
    - N (int): Number of data points.
    - Dn (int): Dimension of the feature vector.
    - true_feature (numpy array): True feature vector.

    Returns:
    - X (numpy array): Design matrix of shape (N, Dn) with synthetic features.
    - y (numpy array): Binary labels (0 or 1) generated using logistic function.
    """

    # Generate random feature matrix X
    X = np.random.binomial(1,0.5, (N,Dn))

    # Generate binary labels using the logistic function
    inner_product = np.dot(X, true_feature)
    prob = 1 / (1 + np.exp(-inner_product))
    y = np.random.binomial(1, prob, size=N)

    return X, y

# Example usage:
# N = 100
# Dn = 10
# true_feature = generate_gaussian_variables(Dn)
# X, y = generate_data(N, Dn, true_feature)


# Example: Generate 10 data points with Dn=5 and true_feature vector
N = 10**6
Dn = 100
true_feature = generate_gaussian_variables(Dn)  # Assuming true_feature is generated using Gaussian distribution

X, y = generate_data(N, Dn, true_feature)
print(N)

# print(f"Generated {N} data points:")
# for i in range(N):
#     print(f"Data Point {i+1}: X={X[i]}, y={y[i]}")
print(f"true feature vector: {true_feature}")

def logit_loss(y_true, y_pred):
    # Compute the logit loss
    loss = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    return np.mean(loss)

def adagrad_optimizer_teacher_logit_with_regret(X, y, teacher_feature_set, learning_rate=0.2, epochs=1, epsilon=1e-8, learning_rate_decay=0.5, has_baseline = False, features_baseline = [], const_baseline = 0):
    N, Dn = X.shape
    true_feature_teacher = np.zeros(Dn)
    true_feature_teacher[teacher_feature_set] = np.random.normal(size = len(teacher_feature_set))
    hidden_feature_value_teacher = np.random.normal()  # Initialize hidden_feature_value_teacher randomly

    G_true_feature_teacher = np.zeros(Dn)  # Initialize squared gradient accumulator for true_feature_teacher
    G_hidden_feature_teacher = 0  # Initialize squared gradient accumulator for hidden_feature_value_teacher

    logit_losses_teacher = []
    regrets = []
    prev_logit_loss_teacher = float('inf')
    regret = 0

    for epoch in range(epochs):
        logit_loss_teacher = 0  # Initialize logit loss for the current epoch for teacher

        for i in range(N):
            # Predict the label using the logistic function of the inner product
            inner_product = np.dot(X[i, teacher_feature_set], true_feature_teacher[teacher_feature_set]) + hidden_feature_value_teacher
            y_pred_teacher = 1 / (1 + np.exp(-inner_product))

            # Compute the gradient of the logit loss with respect to true_feature_teacher and hidden_feature_value_teacher
            gradient_true_feature_teacher = X[i, teacher_feature_set] * (y_pred_teacher - y[i])
            gradient_hidden_feature_teacher = y_pred_teacher - y[i]

            # Update squared gradient accumulators
            G_true_feature_teacher[teacher_feature_set] += gradient_true_feature_teacher**2
            G_hidden_feature_teacher += gradient_hidden_feature_teacher**2

            # Update true_feature_teacher and hidden_feature_value_teacher using AdaGrad update rule
            true_feature_teacher[teacher_feature_set] -= learning_rate / (G_true_feature_teacher[teacher_feature_set] ** learning_rate_decay + epsilon) * gradient_true_feature_teacher
            hidden_feature_value_teacher -= learning_rate / (G_hidden_feature_teacher ** learning_rate_decay + epsilon) * gradient_hidden_feature_teacher

            # Update logit loss for the current epoch for teacher
            logit_loss_teacher += logit_loss(y[i], y_pred_teacher)

            # Calculate regret for the current data point
            true_inner_product = np.dot(X[i], true_feature)
            true_prob = 1 / (1 + np.exp(-true_inner_product))
            if has_baseline:
                baseline_product = np.dot(X[i, teacher_feature_set], features_baseline[teacher_feature_set]) + const_baseline
                baseline_prob = 1/ (1 + np.exp(-baseline_product))
                regret_sample = logit_loss(y[i],y_pred_teacher) - logit_loss(y[i], baseline_prob)
            else:
                regret_sample = logit_loss(y[i],y_pred_teacher) - logit_loss(y[i], true_prob)

            regret += regret_sample
            # regret_epoch += logit_loss(y[i], true_prob)
            regrets.append(regret/(i+1+N*epoch))

            if i%10000==0:
                print(f"sample {i+1}/{N}, Regret: {regret}, Learned Feature: {true_feature_teacher[teacher_feature_set]}, Constant Term: {hidden_feature_value_teacher}")
        logit_losses_teacher.append(logit_loss_teacher)

    # Print learned features for teacher
    # print("\nLearned Features (Teacher's Feature Set):")
    # print(true_feature_teacher[teacher_feature_set])

    return true_feature_teacher, hidden_feature_value_teacher, logit_losses_teacher, regrets

    Class adagrad(X,y, teacher_feature_set, student_feature_set)

