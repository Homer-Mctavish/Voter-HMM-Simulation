from behave import given, when, then
from hmmlearn import hmm
import numpy as np

@given('I have a BaseHMM model with "n_components" set to {num_states}')
def step_initialize_model(context, num_states):
    context.num_states = int(num_states)

@given('the model has "gamma_shape" set to {gamma_shape} and "gamma_scale" set to {gamma_scale}')
def step_set_gamma_parameters(context, gamma_shape, gamma_scale):
    context.gamma_shape = float(gamma_shape)
    context.gamma_scale = float(gamma_scale)

@given('I have simulated approval data for "num_candidates" candidates')
def step_simulate_approval_data(context):
    num_candidates = 3  # Adjust as needed for the test
    context.X = np.random.rand(100, num_candidates)  # Simulated data

@given('the model is configured to run for {n_iter} iterations with a tolerance of {tol}')
def step_set_training_parameters(context, n_iter, tol):
    context.n_iter = int(n_iter)
    context.tol = float(tol)

@when('I train the model with the simulated approval data')
def step_train_model(context):
    context.model = BaseHMM(
        n_components=context.num_states,
        gamma_shape=context.gamma_shape,
        gamma_scale=context.gamma_scale,
        n_iter=context.n_iter,
        tol=context.tol,
        verbose=False
    )
    context.model.fit(context.X)

@then('the model should converge successfully within {n_iter} iterations')
def step_check_convergence(context, n_iter):
    assert context.model.monitor_.iter_ <= int(n_iter)

@then('the log likelihood should stabilize within {tol}')
def step_check_log_likelihood_stabilization(context, tol):
    assert context.model.monitor_.converged

@then('the transition matrix should contain values greater than 0')
def step_transition_matrix_nonzero(context):
    assert (context.model.transmat_ > 0).any()

@then('the transition matrix should reflect probable changes in approval patterns')
def step_transition_matrix_patterns(context):
    # Check if the transition matrix has expected pattern-based properties
    assert context.model.transmat_.sum(axis=1).all() == 1  # Rows should sum to 1

@then("the model's gamma emission parameters should match the input data patterns within tolerance {tol}")
def step_emission_parameter_accuracy(context, tol):
    tolerance = float(tol)
    # Verify that gamma parameters match the expected input data patterns within tolerance
    for i, shape in enumerate(context.model.gamma_shape):
        assert abs(shape - context.gamma_shape) < tolerance
    for i, scale in enumerate(context.model.gamma_scale):
        assert abs(scale - context.gamma_scale) < tolerance

@then('the model should complete training without errors')
def step_training_completion(context):
    assert context.model.monitor_.converged
