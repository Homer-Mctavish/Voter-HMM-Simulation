# 1. Prepare new observation data
X_new = np.array([0.65, 0.7, 0.68, 0.72, 0.75]).reshape(-1, 1)

# 2. Predict hidden states for new data
hidden_states = model.predict(X_new)
print("Predicted Hidden States:")
print(hidden_states)

# 3. Interpret the hidden states
for i in range(model.n_components):
    shape = model.gamma_shape[i]
    scale = model.gamma_scale[i]
    mean = shape * scale
    variance = shape * (scale ** 2)
    print(f"State {i}:")
    print(f"  Shape (k): {shape}")
    print(f"  Scale (θ): {scale}")
    print(f"  Mean Approval Rating: {mean}")
    print(f"  Variance: {variance}\n")

# 4. Compute state probabilities
state_probs = model.predict_proba(X_new)
print("State Probabilities at Each Time Point:")
print(state_probs)

# 5. Predict next state probabilities
current_state_probs = state_probs[-1]
next_state_probs = np.dot(current_state_probs, model.transmat_)
print("Predicted Next State Probabilities:")
print(next_state_probs)

# 6. Expected approval ratings for next states
for i in range(model.n_components):
    shape = model.gamma_shape[i]
    scale = model.gamma_scale[i]
    expected_approval = shape * scale
    state_probability = next_state_probs[i]
    print(f"State {i}:")
    print(f"  Probability: {state_probability}")
    print(f"  Expected Approval Rating: {expected_approval}\n")
