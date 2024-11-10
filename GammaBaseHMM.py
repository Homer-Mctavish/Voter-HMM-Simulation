import numpy as np
from hmmlearn.base import _BaseHMM
from scipy.stats import gamma

class BaseHMM(_BaseHMM):
    def __init__(self, n_components=1, startprob_prior=1.0, transmat_prior=1.0, gamma_shape=1.0, gamma_scale=1.0, n_iter=10, tol=1e-2, verbose=False, random_state=None):
        super(BaseHMM, self).__init__(n_components=n_components, startprob_prior=startprob_prior, transmat_prior=transmat_prior, random_state=random_state, n_iter=n_iter, tol=tol, verbose=verbose)
        
        # Initialize gamma parameters for each state
        self.gamma_shape = np.full(n_components, gamma_shape)  # shape parameter for each state
        self.gamma_scale = np.full(n_components, gamma_scale)  # scale parameter for each state
    
    def _compute_log_likelihood(self, X):
        """
        Compute the log likelihood of each sample in X for each state
        using the gamma emission probabilities.
        """
        log_likelihood = np.empty((X.shape[0], self.n_components))
        
        for i, (shape, scale) in enumerate(zip(self.gamma_shape, self.gamma_scale)):
            # Use gamma distribution's log probability density function
            log_likelihood[:, i] = gamma.logpdf(X, shape, scale=scale)
        
        return log_likelihood
    
    def _initialize_sufficient_statistics(self):
        """
        Initialize the sufficient statistics required for the M-step.
        """
        stats = super()._initialize_sufficient_statistics()
        stats['gamma_shape'] = np.zeros(self.n_components)
        stats['gamma_scale'] = np.zeros(self.n_components)
        return stats
    
    def _accumulate_sufficient_statistics(self, stats, X, framelogprob, posteriors):
        """
        Accumulate sufficient statistics for the gamma distribution parameters.
        """
        super()._accumulate_sufficient_statistics(stats, X, framelogprob, posteriors)
        
        # Update the gamma shape and scale parameters
        for i in range(self.n_components):
            weighted_sum = np.sum(posteriors[:, i] * X)
            weighted_log_sum = np.sum(posteriors[:, i] * np.log(X))
            stats['gamma_shape'][i] += weighted_sum
            stats['gamma_scale'][i] += weighted_log_sum
    
    def _do_mstep(self, stats):
        """
        Perform the M-step to update the model parameters.
        """
        super()._do_mstep(stats)
        
        # Update gamma parameters based on accumulated statistics
        for i in range(self.n_components):
            # Estimate new shape and scale for gamma distribution
            mean_val = stats['gamma_shape'][i] / np.sum(self.transmat_[i])
            var_val = stats['gamma_scale'][i] / np.sum(self.transmat_[i])
            
            # Updating shape and scale based on MLE estimates
            self.gamma_shape[i] = mean_val**2 / var_val
            self.gamma_scale[i] = var_val / mean_val
