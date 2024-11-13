import numpy as np
from pomegranate import HiddenMarkovModel, State
from pomegranate.distributions import IndependentComponentsDistribution, BernoulliDistribution

# Number of candidates
n_candidates = 3  # Adjust based on your data

# Number of hidden states
n_components = 2  # Adjust based on the patterns you expect

# Create the HMM model
model = HiddenMarkovModel(name="ApprovalVotingHMM")

# Initialize states with emission probabilities
states = []
for i in range(n_components):
    # For each state, define a Bernoulli distribution for each candidate
    candidate_distributions = []
    for _ in range(n_candidates):
        # Initialize with random probabilities or prior knowledge
        p = np.random.rand()
        candidate_distributions.append(BernoulliDistribution(p))
    
    # Create an independent components distribution (product of Bernoulli distributions)
    emission_distribution = IndependentComponentsDistribution(candidate_distributions)
    state = State(emission_distribution, name=f"State_{i}")
    states.append(state)
    model.add_state(state)

# Define initial probabilities (uniform)
model.add_transition(model.start, states[0], 0.5)
model.add_transition(model.start, states[1], 0.5)

# Define transition probabilities (initialize randomly)
model.add_transition(states[0], states[0], 0.7)
model.add_transition(states[0], states[1], 0.3)
model.add_transition(states[1], states[0], 0.4)
model.add_transition(states[1], states[1], 0.6)

# Finalize the model structure
model.bake()

# Sample data: list of sequences (we'll assume one sequence here)
X = [
    [1, 0, 1],
    [0, 1, 0],
    [1, 1, 0],
    [1, 0, 0],
    [0, 1, 1],
    # ... more observations
]

# Since we're not dealing with multiple sequences, we can fit directly

# Fit the model to the data using the Baum-Welch algorithm
model.fit(X, algorithm='baum-welch', max_iterations=100)

# Get the hidden state sequence for the observed data
hidden_states = model.predict(X)
last_hidden_state_index = hidden_states[-1]
last_hidden_state = model.states[last_hidden_state_index]

# Get transition probabilities from the last hidden state
transitions = model.transitions[last_hidden_state]
next_state_probs = {}
for transition in transitions:
    if transition.destination != model.end:
        next_state_probs[transition.destination.name] = transition.probability

# For simplicity, assume the most probable next state
next_state_name = max(next_state_probs, key=next_state_probs.get)
next_state = [s for s in model.states if s.name == next_state_name][0]

# Get the emission probabilities (approval probabilities for each candidate)
emission_dists = next_state.distribution.distributions
approval_probs = [dist.parameters[0] for dist in emission_dists]

print(f"Predicted Approval Probabilities for Next Ballot:")
for idx, prob in enumerate(approval_probs):
    print(f"Candidate {idx}: {prob:.2f}")

# Generate a sample from the emission distribution of the next state
predicted_approval = next_state.distribution.sample()
print(f"Predicted Approval Outcome: {predicted_approval}")

# ... [Previous code for model creation and training] ...

# Predict the hidden states for the observed data
hidden_states = model.predict(X)

# Get the last observed hidden state
last_hidden_state_index = hidden_states[-1]
last_hidden_state = model.states[last_hidden_state_index]

# Get possible transitions and their probabilities
transitions = model.transitions[last_hidden_state]
next_state_probs = {}
for transition in transitions:
    if transition.destination != model.end:
        next_state_probs[transition.destination.name] = transition.probability

# Identify the most probable next state
next_state_name = max(next_state_probs, key=next_state_probs.get)
next_state = [s for s in model.states if s.name == next_state_name][0]

# Get emission probabilities for the next state
emission_dists = next_state.distribution.distributions
approval_probs = [dist.parameters[0] for dist in emission_dists]

# Output the predicted approval probabilities
print(f"Predicted Approval Probabilities for Next Ballot:")
for idx, prob in enumerate(approval_probs):
    print(f"Candidate {idx}: {prob:.2f}")

# Optionally, generate a predicted approval outcome
predicted_approval = next_state.distribution.sample()
print(f"Predicted Approval Outcome: {predicted_approval}")
