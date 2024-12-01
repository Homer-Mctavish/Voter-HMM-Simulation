from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from hmmlearn import hmm
import uvicorn

app = FastAPI()

# Define Pydantic models for input data
class Ballot(BaseModel):
    sequence: List[List[int]]  # A sequence of one-hot encoded ballots


class ModelParameters(BaseModel):
    n_components: int  # Number of hidden states
    n_iter: int = 100  # Number of iterations for training


class TrainData(BaseModel):
    sequences: List[List[List[int]]]  # A list of sequences, each sequence is a list of one-hot ballots


# Initialize global model variable
model: Optional[hmm.CategoricalHMM] = None
NUM_CANDIDATES = None  # This will be set with the model parameters


@app.post("/set_parameters/")
def set_parameters(params: ModelParameters, num_candidates: int):
    """
    Endpoint to set the initial parameters of the HMM model.
    """
    global model, NUM_CANDIDATES
    NUM_CANDIDATES = num_candidates
    model = hmm.CategoricalHMM(n_components=params.n_components, n_iter=params.n_iter, random_state=42)
    return {"message": "Model parameters set successfully.", "n_components": params.n_components, "n_iter": params.n_iter}


@app.get("/get_parameters/")
def get_parameters():
    """
    Endpoint to retrieve current model parameters.
    """
    if model is None:
        raise HTTPException(status_code=400, detail="Model parameters have not been set.")
    return {
        "n_components": model.n_components,
        "n_iter": model.n_iter,
        "start_probabilities": model.startprob_.tolist() if hasattr(model, 'startprob_') else None,
        "transition_matrix": model.transmat_.tolist() if hasattr(model, 'transmat_') else None,
        "emission_matrix": model.emissionprob_.tolist() if hasattr(model, 'emissionprob_') else None,
    }


@app.post("/train_model/")
def train_model(train_data: TrainData):
    """
    Endpoint to train the model with sequences of one-hot encoded ballots.
    """
    if model is None or NUM_CANDIDATES is None:
        raise HTTPException(status_code=400, detail="Model parameters have not been set. Please set parameters first.")

    # Prepare training data by concatenating all sequences and storing their lengths
    all_sequences = [np.array(seq) for seq in train_data.sequences]
    lengths = [len(seq) for seq in all_sequences]
    X_train = np.concatenate(all_sequences)

    # Validate that each sequence has the correct one-hot encoding format
    for seq in X_train:
        if len(seq) != NUM_CANDIDATES or sum(seq) != 1:
            raise HTTPException(status_code=400, detail="Each ballot item must be a one-hot encoded vector.")

    # Train the HMM model
    try:
        model.fit(X_train, lengths)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

    return {"message": "Model trained successfully."}


@app.post("/predict_ballot/")
def predict_ballot(ballot: Ballot):
    """
    Endpoint to predict the next approval vector based on a given ballot sequence.
    """
    if model is None:
        raise HTTPException(status_code=400, detail="Model parameters have not been set or model is not trained.")

    # Validate input: each ballot should be one-hot encoded with length of NUM_CANDIDATES
    for item in ballot.sequence:
        if len(item) != NUM_CANDIDATES or sum(item) != 1:
            raise HTTPException(status_code=400, detail="Each ballot item must be a one-hot encoded vector.")

    # Convert input sequence to numpy array
    sequence = np.array(ballot.sequence)

    # Predict hidden states and the next approval vector
    try:
        log_prob, hidden_states = model.decode(sequence, algorithm="viterbi")
        next_state = model.predict(sequence)[-1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    # Convert next state to one-hot encoding
    next_approval_one_hot = [0] * NUM_CANDIDATES
    next_approval_one_hot[next_state] = 1

    return {
        "log_probability": log_prob,
        "hidden_states": hidden_states.tolist(),
        "next_approval_prediction": next_approval_one_hot
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
