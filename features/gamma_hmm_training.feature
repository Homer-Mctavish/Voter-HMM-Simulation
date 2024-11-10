Feature: Training and validating the BaseHMM model with gamma emissions
  In order to ensure that the BaseHMM model functions correctly
  As a data scientist
  I want to test the initialization, training, and output of the model using Cucumber tests

  Background:
    Given I have a BaseHMM model with "n_components" set to <num_states>
    And the model is initialized with gamma shape parameters <gamma_shapes> and gamma scale parameters <gamma_scales>
    And I have sequences of composite scores as observations
    And I have initial transition probabilities set to <initial_transmat>
    And I have initial start probabilities set to <initial_startprob>

  Scenario Outline: Model initialization with gamma parameters
    When I initialize the BaseHMM model
    Then the model's gamma shape parameters should be <gamma_shapes>
    And the model's gamma scale parameters should be <gamma_scales>
    And the model's transition matrix should be <initial_transmat>
    And the model's start probabilities should be <initial_startprob>

    Examples:
      | num_states | gamma_shapes       | gamma_scales       | initial_transmat                   | initial_startprob |
      | 3          | [2.0,2.5,3.0]      | [0.1,0.1,0.1]      | [[0.7,0.2,0.1],[0.1,0.8,0.1],[0.2,0.3,0.5]] | [0.6,0.3,0.1] |

  Scenario: Model training with Baum-Welch algorithm
    When I train the model using the Baum-Welch algorithm
    Then the model should converge within the specified iterations
    And the log likelihood should increase at each iteration

  Scenario: Gamma emission parameters update during training
    When I train the model using the Baum-Welch algorithm
    Then the model's gamma shape parameters should be updated from the initial values
    And the model's gamma scale parameters should be updated from the initial values

  Scenario: Predicting hidden states
    When I use the trained model to predict hidden states
    Then the model should output a sequence of hidden states equal in length to the observations

  Scenario: Transition probabilities update during training
    When I train the model using the Baum-Welch algorithm
    Then the model's transition matrix should be updated from the initial values
    And each row of the transition matrix should sum to 1

  Scenario: Model scoring and log likelihood computation
    When I compute the log likelihood of the observations
    Then the log likelihood should be a finite number

