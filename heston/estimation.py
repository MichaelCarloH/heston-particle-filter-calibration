"""
Parameter estimation methods for Heston model.
"""

import numpy as np
from typing import Dict, Optional, Callable
import logging
from .model import HestonModel, HestonSSM
import particles
from particles import distributions as dists
from particles import state_space_models as ssm
from particles import mcmc
from particles.collectors import Moments
import scipy.optimize

logger = logging.getLogger(__name__)


def diagnose_pmmh_mixing(pmmh_object, burnin=0, verbose=True):
    """
    Diagnose PMMH chain mixing quality.
    
    Parameters
    ----------
    pmmh_object : particles.mcmc.PMMH
        PMMH object after running
    burnin : int
        Burn-in period
    verbose : bool
        Whether to print diagnostics
        
    Returns
    -------
    dict
        Dictionary with diagnostic statistics
    """
    param_names = ['kappa', 'theta', 'sigma', 'rho']
    diagnostics = {}
    
    for param in param_names:
        if param not in pmmh_object.chain.theta.dtype.names:
            continue
            
        samples = pmmh_object.chain.theta[param][burnin:]
        
        diagnostics[param] = {
            'n_samples': len(samples),
            'n_unique': len(np.unique(samples)),
            'mean': samples.mean(),
            'std': samples.std(),
            'min': samples.min(),
            'max': samples.max(),
            'range': samples.max() - samples.min(),
            'is_stuck': len(np.unique(samples)) == 1,
            'mixing_ratio': len(np.unique(samples)) / len(samples) if len(samples) > 0 else 0
        }
        
        if verbose:
            diag = diagnostics[param]
            status = "STUCK" if diag['is_stuck'] else "OK" if diag['mixing_ratio'] > 0.1 else "POOR"
            print(f"{param:8s}: {status:6s} | "
                  f"unique={diag['n_unique']:4d}/{diag['n_samples']:4d} | "
                  f"std={diag['std']:.6f} | "
                  f"range=[{diag['min']:.4f}, {diag['max']:.4f}]")
    
    # Check acceptance rate if available
    if hasattr(pmmh_object, 'acc_rate'):
        diagnostics['acceptance_rate'] = pmmh_object.acc_rate
        if verbose:
            print(f"\nAcceptance rate: {pmmh_object.acc_rate:.2%}")
            if pmmh_object.acc_rate < 0.1:
                print("  WARNING: Very low acceptance rate (<10%). Chain may have mixing issues.")
            elif pmmh_object.acc_rate > 0.5:
                print("  WARNING: Very high acceptance rate (>50%). Proposals may be too small.")
    
    return diagnostics


def estimate_pmmh(
    model: HestonModel,
    n_iter: int = 10000,
    n_particles: int = 200,
    prior: Optional[Dict] = None,
    burnin: int = 1000,
    verbose: bool = True
) -> Dict:
    """
    Estimate parameters using Particle Marginal Metropolis-Hastings (PMMH).
    
    Parameters
    ----------
    model : HestonModel
        Model instance with data loaded
    n_iter : int
        Number of MCMC iterations (default: 10000)
    n_particles : int
        Number of particles per PF run (default: 200)
    prior : dict, optional
        Prior distributions. If None, uses default priors:
        - kappa: Gamma(2.0, 1.0)
        - theta: Gamma(2.0, 40.0)
        - sigma: Gamma(2.0, 2.5)
        - rho: Uniform(-0.99, 0.99)
    burnin : int
        Number of burn-in iterations (default: 1000)
    verbose : bool
        Whether to print progress (default: True)
    
    Returns
    -------
    dict
        Dictionary containing:
        - chain: MCMC chain
        - posterior_means: Posterior mean estimates
        - posterior_stds: Posterior standard deviations
        - credible_intervals: 95% credible intervals
    """
    if model.data is None:
        raise ValueError("No data loaded. Call load_data() first.")
    
    # Default priors
    if prior is None:
        prior = {
            'kappa': dists.Gamma(a=2.0, b=1.0),
            'theta': dists.Gamma(a=2.0, b=40.0),
            'sigma': dists.Gamma(a=2.0, b=2.5),
            'rho': dists.Uniform(a=-0.99, b=0.99)
        }
    
    # Filter out Dirac distributions and fixed parameters from prior
    # Fixed parameters (v0, dt, r) should not be sampled by PMMH
    # Dirac distributions can cause PMMH mixing issues
    if isinstance(prior, dists.StructDist):
        # For StructDist, sample to identify which parameters are fixed (Dirac)
        test_samples = prior.rvs(size=10)
        param_names = list(test_samples.dtype.names)
        
        # Identify fixed parameters (those with zero variance in samples)
        fixed_params = []
        for param in param_names:
            values = test_samples[param]
            if np.allclose(values, values[0], atol=1e-10):
                fixed_params.append(param)
        
        # Filter out fixed parameters and known fixed parameter names
        params_to_keep = [p for p in param_names 
                         if p not in fixed_params and p not in ['v0', 'dt', 'r']]
        
        if len(params_to_keep) < len(param_names):
            if verbose:
                filtered_params = [p for p in param_names if p not in params_to_keep]
                logger.info(f"Filtering out fixed parameters from prior: {filtered_params}")
            
            # Try to access underlying distributions
            # StructDist may have different internal structures, so we try multiple approaches
            prior_dict_filtered = {}
            if hasattr(prior, 'dists'):
                # Direct access to underlying dict
                prior_dict_filtered = {k: v for k, v in prior.dists.items() 
                                      if k in params_to_keep}
            elif hasattr(prior, '_dists'):
                prior_dict_filtered = {k: v for k, v in prior._dists.items() 
                                      if k in params_to_keep}
            
            if prior_dict_filtered:
                prior = dists.StructDist(prior_dict_filtered)
            else:
                # If we can't access the dict directly, warn the user
                if verbose:
                    logger.warning(
                        "Cannot access StructDist internal structure to filter Dirac distributions. "
                        "Using original prior - PMMH may have mixing issues if prior contains Dirac for parameters to estimate."
                    )
                # Keep original prior - user should ensure it doesn't contain Dirac for parameters to estimate
    elif isinstance(prior, dict):
        # If it's a dict, filter out Dirac distributions and fixed parameter names
        prior_for_pmmh = {
            k: v for k, v in prior.items() 
            if not isinstance(v, dists.Dirac) and k not in ['v0', 'dt', 'r']
        }
        if len(prior_for_pmmh) < len(prior):
            if verbose:
                filtered_params = [k for k in prior.keys() if k not in prior_for_pmmh]
                logger.info(f"Filtering out fixed parameters from prior: {filtered_params}")
            prior = prior_for_pmmh
    
    # Create factory function for SSM with fixed risk-free rate
    def create_heston_ssm_class(r_series, dt, v0):
        """Factory function to create HestonSSM class with fixed risk-free rate"""
        class HestonSSM_FixedR(ssm.StateSpaceModel):
            def __init__(self, kappa, theta, sigma, rho, **kwargs):
                super().__init__()
                self.kappa = kappa
                self.theta = theta
                self.sigma = sigma
                self.rho = rho
                self.r = r_series  # Fixed time series
                self.dt = dt
                self.v0 = v0
            
            def PX0(self):
                return dists.Dirac(self.v0)
            
            def PX(self, t, xp):
                vp = np.maximum(np.asarray(xp), 1e-12)
                mean = vp + self.kappa * (self.theta - vp) * self.dt
                std = self.sigma * np.sqrt(vp * self.dt)
                return dists.Normal(loc=mean, scale=std)
            
            def PY(self, t, xp, x):
                # Use V_{t-1} (xp) to match paper's discretization
                # Handle initial observation (t=0): use V_0 (x) when xp is None
                if xp is None:
                    vp = np.maximum(np.asarray(x), 1e-12)  # Use V_0 at t=0
                else:
                    vp = np.maximum(np.asarray(xp), 1e-12)  # Use V_{t-1} for t>0
                r_t = self.r[t] if hasattr(self.r, '__len__') and len(self.r) > 1 else self.r
                mean = (r_t - 0.5 * vp) * self.dt
                std = np.sqrt(vp * self.dt)
                return dists.Normal(loc=mean, scale=std)
        
        return HestonSSM_FixedR
    
    # Create model class with fixed risk-free rate
    HestonSSM_for_PMMH = create_heston_ssm_class(
        r_series=model.r,
        dt=model.dt,
        v0=model.v0
    )
    
    # Validate burn-in
    if burnin >= n_iter:
        raise ValueError(
            f"burnin ({burnin}) must be less than n_iter ({n_iter}). "
            f"After burn-in, there would be no samples left."
        )
    
    if verbose:
        logger.info(f"Running PMMH with {n_iter} iterations...")
        logger.info(f"Using {n_particles} particles per PF run")
        logger.info(f"Data length: {len(model.data)} observations")
        logger.info(f"Burn-in: {burnin} iterations (will use {n_iter - burnin} samples for posterior)")
        # Rough estimate: each iteration processes ~len(data) observations with n_particles
        # This is a very rough estimate - actual time depends on many factors
        estimated_minutes = max(5, n_iter * len(model.data) * n_particles / 5e7)
        logger.info(f"Estimated time: ~{estimated_minutes:.0f}-{estimated_minutes*2:.0f} minutes (rough estimate)")
        logger.info("Note: PMMH runs silently without progress updates - this is normal.")
        logger.info("      The particles library doesn't provide progress bars for PMMH.")
        if n_iter >= 10000:
            logger.warning("For faster testing, consider reducing n_iter to 1000-2000 iterations")
    
    # Run PMMH
    pmmh = mcmc.PMMH(
        ssm_cls=HestonSSM_for_PMMH,
        prior=prior,
        data=model.data,
        Nx=n_particles,
        niter=n_iter
    )
    
    if verbose:
        logger.info("Starting PMMH...")
    
    pmmh.run()
    
    if verbose:
        logger.info("PMMH completed!")
    
    # Extract posterior statistics
    param_names = ['kappa', 'theta', 'sigma', 'rho']
    posterior_means = {}
    posterior_stds = {}
    credible_intervals = {}
    mixing_warnings = []
    
    for param in param_names:
        samples = pmmh.chain.theta[param][burnin:]
        
        if len(samples) == 0:
            raise ValueError(
                f"No samples remaining after burn-in. "
                f"burnin={burnin}, n_iter={n_iter}. "
                f"Reduce burnin or increase n_iter."
            )
        
        # Check for stuck chain
        unique_values = len(np.unique(samples))
        sample_range = samples.max() - samples.min()
        sample_mean = samples.mean()
        
        posterior_means[param] = sample_mean
        posterior_stds[param] = samples.std()
        
        # Diagnose mixing issues
        if unique_values == 1 or sample_range < 1e-10:
            mixing_warnings.append(
                f"Chain appears STUCK for '{param}': all {len(samples)} samples identical "
                f"(value={samples[0]:.6f})"
            )
        elif abs(sample_mean) > 1e-6 and sample_range < 0.01 * abs(sample_mean):
            mixing_warnings.append(
                f"Chain may have POOR MIXING for '{param}': range ({sample_range:.6f}) is very small "
                f"relative to mean ({sample_mean:.6f})"
            )
        
        # For credible intervals, need at least 2 samples
        if len(samples) >= 2:
            credible_intervals[param] = (
                np.percentile(samples, 2.5),
                np.percentile(samples, 97.5)
            )
        else:
            # If only 1 sample, use that value for both bounds
            credible_intervals[param] = (samples[0], samples[0])
    
    # Print mixing warnings
    if mixing_warnings and verbose:
        logger.warning("\n" + "="*60)
        logger.warning("PMMH MIXING WARNINGS:")
        logger.warning("="*60)
        for warning in mixing_warnings:
            logger.warning(f"  ⚠ {warning}")
        logger.warning("\nSuggestions to improve mixing:")
        logger.warning(f"  1. Increase n_particles (currently {n_particles}) - try 500-1000")
        logger.warning(f"  2. Increase n_iter (currently {n_iter}) - try 20000+")
        logger.warning(f"  3. Check if priors are too restrictive")
        logger.warning(f"  4. Try different initial values")
        logger.warning("="*60 + "\n")
    
    if verbose:
        logger.info("\n=== PMMH Posterior Statistics ===")
        for param in param_names:
            logger.info(
                f"{param:8s}: mean={posterior_means[param]:.4f}, "
                f"std={posterior_stds[param]:.4f}, "
                f"95% CI=[{credible_intervals[param][0]:.4f}, "
                f"{credible_intervals[param][1]:.4f}]"
            )
    
    return {
        'chain': pmmh.chain,
        'posterior_means': posterior_means,
        'posterior_stds': posterior_stds,
        'credible_intervals': credible_intervals,
        'pmmh_object': pmmh,
        'mixing_warnings': mixing_warnings
    }


def estimate_simplex(
    model: HestonModel,
    initial_params: Optional[Dict] = None,
    bounds: Optional[Dict] = None,
    n_particles: int = 2000,
    verbose: bool = True
) -> Dict:
    """
    Estimate parameters using Nelder-Mead simplex method.
    
    Parameters
    ----------
    model : HestonModel
        Model instance with data loaded
    initial_params : dict, optional
        Initial parameter values. If None, uses defaults:
        - kappa: 2.0
        - theta: 0.04
        - sigma: 0.3
        - rho: -0.7
    bounds : dict, optional
        Parameter bounds as (min, max) tuples
    n_particles : int
        Number of particles for likelihood estimation (default: 2000)
    verbose : bool
        Whether to print progress (default: True)
    
    Returns
    -------
    dict
        Dictionary containing:
        - estimated_params: Estimated parameter values
        - log_likelihood: Final log-likelihood
        - optimization_result: scipy optimization result
    """
    if model.data is None:
        raise ValueError("No data loaded. Call load_data() first.")
    
    # Default initial parameters
    if initial_params is None:
        initial_params = {
            'kappa': 2.0,
            'theta': 0.04,
            'sigma': 0.3,
            'rho': -0.7
        }
    
    # Default bounds
    if bounds is None:
        bounds = {
            'kappa': (0.1, 10.0),
            'theta': (0.001, 0.5),
            'sigma': (0.01, 2.0),
            'rho': (-0.99, 0.99)
        }
    
    param_names = ['kappa', 'theta', 'sigma', 'rho']
    x0 = np.array([initial_params[p] for p in param_names])
    param_bounds = [bounds[p] for p in param_names]
    
    def negative_log_likelihood(params):
        """Negative log-likelihood function for optimization"""
        kappa, theta, sigma, rho = params
        
        # Create SSM
        ssm_model = HestonSSM(
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            rho=rho,
            r=model.r,
            dt=model.dt,
            v0=model.v0
        )
        
        # Run particle filter
        fk = ssm.Bootstrap(ssm=ssm_model, data=model.data)
        alg = particles.SMC(fk=fk, N=n_particles, resampling='systematic')
        alg.run()
        
        return -alg.logLt  # Negative for minimization
    
    if verbose:
        logger.info("Running Nelder-Mead optimization...")
    
    result = scipy.optimize.minimize(
        negative_log_likelihood,
        x0,
        method='Nelder-Mead',
        bounds=param_bounds,
        options={'maxiter': 100, 'disp': verbose}
    )
    
    estimated_params = {param: result.x[i] for i, param in enumerate(param_names)}
    
    if verbose:
        logger.info("\n=== Simplex Optimization Results ===")
        for param, value in estimated_params.items():
            logger.info(f"{param:8s}: {value:.4f}")
        logger.info(f"Log-likelihood: {-result.fun:.2f}")
    
    return {
        'estimated_params': estimated_params,
        'log_likelihood': -result.fun,
        'optimization_result': result
    }


def estimate_em(
    model: HestonModel,
    initial_params: Optional[Dict] = None,
    n_iter: int = 20,
    n_particles: int = 2000,
    tolerance: float = 1e-4,
    verbose: bool = True
) -> Dict:
    """
    Estimate parameters using Expectation-Maximization (EM) algorithm.
    
    For state-space models, EM involves:
    - E-step: Run particle filter/smoother to compute expected sufficient statistics
    - M-step: Update parameters using these statistics
    
    Note: For the Heston model, the M-step is approximated using numerical optimization
    since the parameters appear non-linearly.
    
    Parameters
    ----------
    model : HestonModel
        Model instance with data loaded
    initial_params : dict, optional
        Initial parameter values. If None, uses defaults:
        - kappa: 2.0
        - theta: 0.04
        - sigma: 0.3
        - rho: -0.7
    n_iter : int
        Number of EM iterations (default: 20)
    n_particles : int
        Number of particles for particle filter (default: 2000)
    tolerance : float
        Convergence tolerance (default: 1e-4)
    verbose : bool
        Whether to print progress (default: True)
    
    Returns
    -------
    dict
        Dictionary containing:
        - estimated_params: Estimated parameter values
        - log_likelihood: Final log-likelihood
        - iterations: Number of iterations run
        - convergence_history: Parameter values at each iteration
    """
    if model.data is None:
        raise ValueError("No data loaded. Call load_data() first.")
    
    # Default initial parameters
    if initial_params is None:
        initial_params = {
            'kappa': 2.0,
            'theta': 0.04,
            'sigma': 0.3,
            'rho': -0.7
        }
    
    param_names = ['kappa', 'theta', 'sigma', 'rho']
    current_params = initial_params.copy()
    convergence_history = {'log_likelihood': [], 'params': []}
    
    if verbose:
        logger.info(f"Running EM algorithm with {n_iter} iterations...")
        logger.info(f"Initial parameters: {current_params}")
    
    # EM iterations
    for iteration in range(n_iter):
        # E-step: Run particle filter with current parameters
        ssm_model = HestonSSM(
            kappa=current_params['kappa'],
            theta=current_params['theta'],
            sigma=current_params['sigma'],
            rho=current_params['rho'],
            r=model.r,
            dt=model.dt,
            v0=model.v0
        )
        
        fk = ssm.Bootstrap(ssm=ssm_model, data=model.data)
        alg = particles.SMC(fk=fk, N=n_particles, resampling='systematic', collect=[Moments()])
        alg.run()
        
        current_log_likelihood = alg.logLt
        convergence_history['log_likelihood'].append(current_log_likelihood)
        convergence_history['params'].append(current_params.copy())
        
        if verbose:
            logger.info(f"Iteration {iteration+1}/{n_iter}: log-likelihood = {current_log_likelihood:.2f}")
        
        # M-step: Update parameters
        # For Heston model, we use a simplified M-step based on filtered moments
        # This is an approximation - full EM would require particle smoother
        
        # Extract filtered variance moments
        filtered_variance = np.array([m['mean'] for m in alg.summaries.moments]).flatten()
        
        # Approximate M-step using moment matching
        # This is a simplified approach - full EM would require backward smoothing
        # We update parameters based on filtered variance statistics
        
        # Update theta (long-run variance) as mean of filtered variance
        new_theta = np.mean(filtered_variance)
        
        # Update kappa based on variance of filtered variance (mean reversion)
        # This is a heuristic - full EM would require more sophisticated updates
        variance_of_variance = np.var(filtered_variance)
        if variance_of_variance > 0:
            # Rough estimate: higher variance -> lower kappa (slower mean reversion)
            new_kappa = current_params['kappa'] * (1 - 0.1 * min(1.0, variance_of_variance / new_theta))
            new_kappa = max(0.1, min(10.0, new_kappa))  # Keep in reasonable bounds
        else:
            new_kappa = current_params['kappa']
        
        # For sigma and rho, use a simple gradient step based on log-likelihood
        # In practice, these would require more sophisticated updates
        # We'll use a small random walk or keep them fixed for now
        # A better approach would be to use numerical optimization in M-step
        
        # Simple update: try small perturbations and keep if log-likelihood improves
        best_params = current_params.copy()
        best_ll = current_log_likelihood
        
        # Try updating sigma
        for sigma_perturb in [0.9, 1.0, 1.1]:
            test_sigma = current_params['sigma'] * sigma_perturb
            test_sigma = max(0.01, min(2.0, test_sigma))
            
            test_ssm = HestonSSM(
                kappa=new_kappa,
                theta=new_theta,
                sigma=test_sigma,
                rho=current_params['rho'],
                r=model.r,
                dt=model.dt,
                v0=model.v0
            )
            test_fk = ssm.Bootstrap(ssm=test_ssm, data=model.data)
            test_alg = particles.SMC(fk=test_fk, N=n_particles//4, resampling='systematic')
            test_alg.run()
            
            if test_alg.logLt > best_ll:
                best_ll = test_alg.logLt
                best_params['sigma'] = test_sigma
        
        # Try updating rho
        for rho_perturb in [-0.1, 0.0, 0.1]:
            test_rho = current_params['rho'] + rho_perturb
            test_rho = max(-0.99, min(0.99, test_rho))
            
            test_ssm = HestonSSM(
                kappa=new_kappa,
                theta=new_theta,
                sigma=best_params['sigma'],
                rho=test_rho,
                r=model.r,
                dt=model.dt,
                v0=model.v0
            )
            test_fk = ssm.Bootstrap(ssm=test_ssm, data=model.data)
            test_alg = particles.SMC(fk=test_fk, N=n_particles//4, resampling='systematic')
            test_alg.run()
            
            if test_alg.logLt > best_ll:
                best_ll = test_alg.logLt
                best_params['rho'] = test_rho
        
        # Update parameters
        best_params['kappa'] = new_kappa
        best_params['theta'] = new_theta
        
        # Check convergence
        param_change = max(
            abs(best_params['kappa'] - current_params['kappa']),
            abs(best_params['theta'] - current_params['theta']),
            abs(best_params['sigma'] - current_params['sigma']),
            abs(best_params['rho'] - current_params['rho'])
        )
        
        current_params = best_params
        
        if param_change < tolerance:
            if verbose:
                logger.info(f"Converged after {iteration+1} iterations (change: {param_change:.6f})")
            break
    
    # Final evaluation with full particle count
    final_ssm = HestonSSM(
        kappa=current_params['kappa'],
        theta=current_params['theta'],
        sigma=current_params['sigma'],
        rho=current_params['rho'],
        r=model.r,
        dt=model.dt,
        v0=model.v0
    )
    final_fk = ssm.Bootstrap(ssm=final_ssm, data=model.data)
    final_alg = particles.SMC(fk=final_fk, N=n_particles, resampling='systematic')
    final_alg.run()
    final_log_likelihood = final_alg.logLt
    
    if verbose:
        logger.info("\n=== EM Algorithm Results ===")
        for param, value in current_params.items():
            logger.info(f"{param:8s}: {value:.4f}")
        logger.info(f"Final log-likelihood: {final_log_likelihood:.2f}")
        logger.info(f"Iterations: {len(convergence_history['log_likelihood'])}")
    
    return {
        'estimated_params': current_params,
        'log_likelihood': final_log_likelihood,
        'iterations': len(convergence_history['log_likelihood']),
        'convergence_history': convergence_history
    }

