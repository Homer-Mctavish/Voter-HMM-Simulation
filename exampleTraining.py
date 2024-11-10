import numpy as np
from your_module import BaseHMM  # Replace with the actual module name

# Prepare data
sequence1 = np.array([0.5, 0.6, 0.55, 0.7])
sequence2 = np.array([0.4, 0.45, 0.5])
sequence3 = np.array([0.65, 0.6, 0.7, 0.75, 0.8])
X_sequences = [sequence1, sequence2, sequence3]
X = np.concatenate(X_sequences)
lengths = [len(seq) for seq in X_sequences]

# Initialize model
num_states = 3
initial_gamma_shapes = np.array([2.0, 2.5, 3.0])
initial_gamma_scales = np.array([0.1, 0.1, 0.1])
model = BaseHMM(
    n_components=num_states,
    gamma_shape=initial_gamma_shapes,
    gamma_scale=initial_gamma_scales,
    n_iter=100,
    tol=1e-4,
    verbose=True
)

# Set initial transition and start probabilities if available
initial_transition_matrix = np.array([
    [0.7, 0.2, 0.1],
    [0.1, 0.8, 0.1],
    [0.2, 0.3, 0.5]
])
model.transmat_ = initial_transition_matrix

initial_startprob = np.array([0.6, 0.3, 0.1])
model.startprob_ = initial_startprob

# Fit the model
model.fit(X.reshape(-1, 1), lengths)

# Inspect the learned parameters
print("Transition Matrix:")
print(model.transmat_)

print("\nStart Probabilities:")
print(model.startprob_)

print("\nGamma Shape Parameters:")
print(model.gamma_shape)

print("\nGamma Scale Parameters:")
print(model.gamma_scale)

# Predict hidden states
hidden_states = model.predict(X.reshape(-1, 1), lengths)
print("\nHidden States:")
print(hidden_states)

# Compute log likelihood
log_likelihood = model.score(X.reshape(-1, 1), lengths)
print("\nLog Likelihood of the Data:")
print(log_likelihood)
