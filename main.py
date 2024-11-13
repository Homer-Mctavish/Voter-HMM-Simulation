from fastapi import FastAPI, HTTPException
from pomegranate import HiddenMarkovModel, State
from pomegranate.distributions import IndependentComponentsDistribution, BernoulliDistribution
import numpy as np
from typing import List
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Create the FastAPI app
app = FastAPI()

# Allow CORS for all origins (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold the model and its parameters
model = None
n_candidates = None
n_states = None

# Data models for request bodies
class ModelParameters(BaseModel):
    n_candidates: int
    n_states: int
    initial_probabilities: List[float] = None  # Optional
    transition_probabilities: List[List[float]] = None  # Optional
    emission_probabilities: List[List[float]] = None  # Optional

class Observations(BaseModel):
    data: List[List[int]]  # List of observations (each is a list of 0s and 1s)

@app.post("/initialize_model")
def initialize_model(params: ModelParameters):
    global model, n_candidates, n_states

    n_candidates = params.n_candidates
    n_states = params.n_states

    # Validate initial probabilities
    if params.initial_probabilities:
        if len(params.initial_probabilities) != n_states:
            raise HTTPException(status_code=400, detail="Length of initial_probabilities must equal n_states")
        initial_probabilities = params.initial_probabilities
    else:
        # Default to uniform distribution
        initial_probabilities = [1.0 / n_states] * n_states

    # Validate transition probabilities
    if params.transition_probabilities:
        if len(params.transition_probabilities) != n_states or any(len(row) != n_states for row in params.transition_probabilities):
            raise HTTPException(status_code=400, detail="transition_probabilities must be a square matrix of size n_states")
        transition_probabilities = params.transition_probabilities
    else:
        # Initialize randomly and normalize
        transition_probabilities = np.random.rand(n_states, n_states)
        transition_probabilities /= transition_probabilities.sum(axis=1, keepdims=True)
        transition_probabilities = transition_probabilities.tolist()

    # Validate emission probabilities
    if params.emission_probabilities:
        if len(params.emission_probabilities) != n_states or any(len(ep) != n_candidates for ep in params.emission_probabilities):
            raise HTTPException(status_code=400, detail="emission_probabilities must have shape (n_states, n_candidates)")
        emission_probabilities = params.emission_probabilities
    else:
        # Initialize randomly
        emission_probabilities = np.random.rand(n_states, n_candidates).tolist()

    # Create the HMM model
    model = HiddenMarkovModel(name="ApprovalVotingHMM")

    # Create states with emission distributions
    states = []
    for i in range(n_states):
        candidate_distributions = []
        for p in emission_probabilities[i]:
            # Ensure p is between 0 and 1
            p = max(0.0, min(1.0, p))
            candidate_distributions.append(BernoulliDistribution(p))
        emission_distribution = IndependentComponentsDistribution(candidate_distributions)
        state = State(emission_distribution, name=f"State_{i}")
        states.append(state)
        model.add_state(state)

    # Add transitions from the start state
    for i, prob in enumerate(initial_probabilities):
        model.add_transition(model.start, states[i], prob)

    # Add transitions between states
    for i in range(n_states):
        for j in range(n_states):
            prob = transition_probabilities[i][j]
            model.add_transition(states[i], states[j], prob)

    # Finalize the model structure
    model.bake()

    return {"message": "Model initialized successfully"}

@app.post("/train_model")
def train_model(observations: Observations):
    global model
    if model is None:
        raise HTTPException(status_code=400, detail="Model has not been initialized")

    X = observations.data

    # Validate observations
    for obs in X:
        if len(obs) != n_candidates:
            raise HTTPException(status_code=400, detail=f"Each observation must have length n_candidates ({n_candidates})")
        if not all(o in [0, 1] for o in obs):
            raise HTTPException(status_code=400, detail="Observations must be lists of 0s and 1s")

    # Train the model using the Baum-Welch algorithm
    try:
        model.fit(X, algorithm='baum-welch', max_iterations=100)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during training: {str(e)}")

    return {"message": "Model trained successfully"}

@app.post("/predict")
def predict_next_observation():
    global model
    if model is None:
        raise HTTPException(status_code=400, detail="Model has not been initialized")

    # Since we need to predict the next state's emission probabilities, we use the model's transition probabilities
    # For simplicity, we start from the initial state distribution
    initial_probs = {state.name: transition.probability for transition in model.transitions[model.start]}

    # Identify the most probable starting state
    start_state_name = max(initial_probs, key=initial_probs.get)
    start_state = next(state for state in model.states if state.name == start_state_name)

    # Get possible transitions from the starting state
    transitions = model.transitions[start_state]
    next_state_probs = {}
    for transition in transitions:
        if transition.destination != model.end:
            next_state_probs[transition.destination.name] = transition.probability

    # Identify the most probable next state
    next_state_name = max(next_state_probs, key=next_state_probs.get)
    next_state = next(state for state in model.states if state.name == next_state_name)

    # Get emission probabilities for the next state
    emission_dists = next_state.distribution.distributions
    approval_probs = [dist.parameters[0] for dist in emission_dists]

    # Generate a predicted approval outcome
    predicted_approval = next_state.distribution.sample()

    return {
        "predicted_state": next_state.name,
        "approval_probabilities": approval_probs,
        "predicted_approval": predicted_approval,
    }

@app.get("/get_model_parameters")
def get_model_parameters():
    global model
    if model is None:
        raise HTTPException(status_code=400, detail="Model has not been initialized")

    # Get initial probabilities
    initial_probabilities = {}
    for transition in model.transitions[model.start]:
        if transition.destination != model.end:
            initial_probabilities[transition.destination.name] = transition.probability

    # Get transition probabilities
    transition_probabilities = {}
    for state in model.states:
        if state in (model.start, model.end):
            continue
        transitions = model.transitions[state]
        transition_probs = {}
        for transition in transitions:
            if transition.destination != model.end:
                transition_probs[transition.destination.name] = transition.probability
        transition_probabilities[state.name] = transition_probs

    # Get emission probabilities
    emission_probabilities = {}
    for state in model.states:
        if state in (model.start, model.end):
            continue
        emission_dists = state.distribution.distributions
        probs = [dist.parameters[0] for dist in emission_dists]
        emission_probabilities[state.name] = probs

    return {
        "n_candidates": n_candidates,
        "n_states": n_states,
        "initial_probabilities": initial_probabilities,
        "transition_probabilities": transition_probabilities,
        "emission_probabilities": emission_probabilities,
    }
