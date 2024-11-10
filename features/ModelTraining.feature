Feature: Training the BaseHMM model with various configurations
  In order to verify that the BaseHMM model trains successfully
  As a data scientist
  I want to ensure that the model converges, learns state transitions, and correctly configures emissions

  Background:
    Given I have a BaseHMM model with "n_components" set to <num_states>
    And the model has "gamma_shape" set to <gamma_shape> and "gamma_scale" set to <gamma_scale>
    And I have simulated approval data for "num_candidates" candidates

  Scenario Outline: Training the model to successful convergence
    Given the model is configured to run for <n_iter> iterations with a tolerance of <tol>
    When I train the model with the simulated approval data
    Then the model should converge successfully within <n_iter> iterations
    And the log likelihood should stabilize within <tol>

    Examples:
      | num_states | gamma_shape | gamma_scale | n_iter | tol |
      | 3          | 2.0         | 0.1        | 50     | 1e-4 |
      | 4          | 2.5         | 0.2        | 100    | 1e-3 |
      | 5          | 3.0         | 0.15       | 150    | 1e-5 |

  Scenario Outline: Verifying non-trivial transition matrix after training
    Given the model is configured to have "n_components" set to <num_states>
    When I train the model with the simulated approval data
    Then the transition matrix should contain values greater than 0
    And the transition matrix should reflect probable changes in approval patterns

    Examples:
      | num_states |
      | 3          |
      | 5          |
      | 7          |

  Scenario Outline: Checking emission distributions for gamma-based patterns
    Given I have generated synthetic approval data that fits gamma distributions
    And the model is configured with gamma emission probabilities
    When I train the model with the synthetic data
    Then the model's gamma emission parameters should match the input data patterns within tolerance <tol>

    Examples:
      | num_states | gamma_shape | gamma_scale | tol   |
      | 3          | 1.5         | 0.1        | 0.05  |
      | 4          | 2.0         | 0.2        | 0.03  |
      | 5          | 2.5         | 0.15       | 0.02  |

  Scenario Outline: Testing model robustness with different approval patterns
    Given I have approval data that represents diverse patterns for "num_candidates" candidates
    And the model is initialized with "n_components" as <num_states>
    When I train the model on this diverse data
    Then the trained model should classify the data into distinct states

    Examples:
      | num_states |
      | 3          |
      | 5          |
      | 6          |

  Scenario Outline: Model should handle large number of iterations without failing
    Given I have a large dataset of simulated approval ratings
    And the model is configured for <n_iter> iterations with a tolerance of <tol>
    When I train the model with this large dataset
    Then the model should complete training without errors
    And it should converge within <n_iter> iterations

    Examples:
      | num_states | n_iter | tol   |
      | 3          | 200    | 1e-4  |
      | 4          | 300    | 1e-5  |
      | 5          | 500    | 1e-6  |
