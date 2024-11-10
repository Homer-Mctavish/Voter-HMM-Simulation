from hmmlearn import hmm
import numpy as np

# Assuming you have sequences of composite scores
X_sequences = [sequence1, sequence2, sequence3, ...]  # List of sequences (arrays)

# Flatten sequences and compute lengths
X = np.concatenate(X_sequences)
lengths = [len(seq) for seq in X_sequences]

# Initialize the HMM
model = BaseHMM(
    n_components=num_states,
    n_iter=100,
    tol=1e-4,
    verbose=True
)

# Set initial emission parameters (gamma distributions)
model.gamma_shape = initial_gamma_shapes  # Array of shape parameters for each state
model.gamma_scale = initial_gamma_scales  # Array of scale parameters for each state

# Optionally, set initial transition probabilities
model.transmat_ = initial_transition_matrix  # num_states x num_states matrix

# Fit the model to the data
model.fit(X, lengths)
