# features/steps/gamma_hmm_steps.py

from behave import given, when, then
import numpy as np
from your_module import BaseHMM  # Replace 'your_module' with the actual module name

@given('I have a BaseHMM model with "n_components" set to {num_states}')
def step_set_num_states(context, num_states):
    context.num_states = int(num_states)

@given('the model is initialized with gamma shape parameters {gamma_shapes} and gamma scale parameters {gamma_scales}')
def step_set_gamma_parameters(context, gamma_shapes, gamma_scales):
    context.gamma_shapes = np.fromstring(gamma_shapes.strip('[]'), sep=',')
    context.gamma_scales = np.fromstring(gamma_scales.strip('[]'), sep=',')

@given('I have sequences of composite scores as observations')
def step_create_observations(context):
    # Create example sequences as in the script
    sequence1 = np.array([0.5, 0.6, 0.55, 0.7])
    sequence2 = np.array([0.4, 0.45, 0.5])
    sequence3 = np.array([0.65, 0.6, 0.7, 0.75, 0.8])
    context.X_sequences = [sequence1, sequence2, sequence3]
    context.X = np.concatenate(context.X_sequences)
    context.lengths = [len(seq) for seq in context.X_sequences]

@given('I have initial transition probabilities set to {initial_transmat}')
def step_set_initial_transmat(context, initial_transmat):
    transmat_list = eval(initial_transmat)
    context.initial_transmat = np.array(transmat_list)

@given('I have initial start probabilities set to {initial_startprob}')
def step_set_initial_startprob(context, initial_startprob):
    context.initial_startprob = np.fromstring(initial_startprob.strip('[]'), sep=',')

@when('I initialize the BaseHMM model')
def step_initialize_model(context):
    context.model = BaseHMM(
        n_components=context.num_states,
        gamma_shape=context.gamma_shapes,
        gamma_scale=context.gamma_scales,
        n_iter=100,
        tol=1e-4,
        verbose=False
    )
    # Set initial transition and start probabilities
    context.model.transmat_ = context.initial_transmat
    context.model.startprob_ = context.initial_startprob

@then('the model\'s gamma shape parameters should be {gamma_shapes}')
def step_check_gamma_shape_params(context, gamma_shapes):
    expected_shapes = np.fromstring(gamma_shapes.strip('[]'), sep=',')
    np.testing.assert_array_almost_equal(context.model.gamma_shape, expected_shapes)

@then('the model\'s gamma scale parameters should be {gamma_scales}')
def step_check_gamma_scale_params(context, gamma_scales):
    expected_scales = np.fromstring(gamma_scales.strip('[]'), sep=',')
    np.testing.assert_array_almost_equal(context.model.gamma_scale, expected_scales)

@then('the model\'s transition matrix should be {initial_transmat}')
def step_check_transition_matrix(context, initial_transmat):
    expected_transmat = np.array(eval(initial_transmat))
    np.testing.assert_array_almost_equal(context.model.transmat_, expected_transmat)

@then('the model\'s start probabilities should be {initial_startprob}')
def step_check_start_probabilities(context, initial_startprob):
    expected_startprob = np.fromstring(initial_startprob.strip('[]'), sep=',')
    np.testing.assert_array_almost_equal(context.model.startprob_, expected_startprob)

@when('I train the model using the Baum-Welch algorithm')
def step_train_model(context):
    context.model.fit(context.X.reshape(-1, 1), context.lengths)

@then('the model should converge within the specified iterations')
def step_check_convergence(context):
    assert context.model.monitor_.converged, "Model did not converge"

@then('the log likelihood should increase at each iteration')
def step_check_log_likelihood_increasing(context):
    log_likelihoods = context.model.monitor_.history
    assert all(x < y for x, y in zip(log_likelihoods, log_likelihoods[1:])), "Log likelihood did not increase monotonically"

@then('the model\'s gamma shape parameters should be updated from the initial values')
def step_check_gamma_shape_updated(context):
    updated_shapes = context.model.gamma_shape
    assert not np.array_equal(updated_shapes, context.gamma_shapes), "Gamma shape parameters did not update"

@then('the model\'s gamma scale parameters should be updated from the initial values')
def step_check_gamma_scale_updated(context):
    updated_scales = context.model.gamma_scale
    assert not np.array_equal(updated_scales, context.gamma_scales), "Gamma scale parameters did not update"

@when('I use the trained model to predict hidden states')
def step_predict_hidden_states(context):
    context.hidden_states = context.model.predict(context.X.reshape(-1, 1), context.lengths)

@then('the model should output a sequence of hidden states equal in length to the observations')
def step_check_hidden_states_length(context):
    assert len(context.hidden_states) == len(context.X), "Hidden states length does not match observations"

@then('the model\'s transition matrix should be updated from the initial values')
def step_check_transition_matrix_updated(context):
    updated_transmat = context.model.transmat_
    assert not np.array_equal(updated_transmat, context.initial_transmat), "Transition matrix did not update"

@then('each row of the transition matrix should sum to 1')
def step_check_transition_matrix_rowsum(context):
    row_sums = context.model.transmat_.sum(axis=1)
    np.testing.assert_array_almost_equal(row_sums, np.ones(context.num_states))

@when('I compute the log likelihood of the observations')
def step_compute_log_likelihood(context):
    context.log_likelihood = context.model.score(context.X.reshape(-1, 1), context.lengths)

@then('the log likelihood should be a finite number')
def step_check_log_likelihood_finite(context):
    assert np.isfinite(context.log_likelihood), "Log likelihood is not finite"
