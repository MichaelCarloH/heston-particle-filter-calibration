"""
Heston State-Space Model implementation.
"""

import numpy as np
from typing import Optional, Union, List
import particles
from particles import distributions as dists
from particles import state_space_models as ssm
from particles.collectors import Moments
import logging

logger = logging.getLogger(__name__)


class HestonSSM(ssm.StateSpaceModel):
    """
    Heston Stochastic Volatility Model as a State-Space Model.
    
    The Heston model is discretized using Euler scheme:
    - State: V_t (variance)
    - Observation: R_t (log returns)
    
    Parameters
    ----------
    kappa : float
        Mean reversion speed
    theta : float
        Long-run variance
    sigma : float
        Volatility of volatility
    rho : float
        Correlation between asset and variance Brownian motions
        (Note: not used in this discretization, but kept for compatibility)
    r : float or array-like
        Risk-free rate (constant or time-varying)
    dt : float
        Time step (default: 1/252 for daily data)
    v0 : float
        Initial variance
    """
    
    def __init__(
        self,
        kappa: float,
        theta: float,
        sigma: float,
        rho: float,
        r: Union[float, np.ndarray],
        dt: float = 1/252,
        v0: float = 0.04
    ):
        super().__init__()
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.r = r  # Can be scalar (constant) or array (time-varying)
        self.dt = dt
        self.v0 = v0

    def PX0(self):
        """Initial distribution of variance V_0."""
        return dists.Dirac(self.v0)

    def PX(self, t, xp):
        """
        Transition distribution: V_t | V_{t-1} = xp.
        
        Uses Euler discretization of CIR process:
        V_t = V_{t-1} + κ(θ - V_{t-1})Δt + σ√(V_{t-1}Δt) ε_t
        """
        vp = np.maximum(np.asarray(xp), 1e-12)  # Ensure positive
        mean = vp + self.kappa * (self.theta - vp) * self.dt
        std = self.sigma * np.sqrt(vp * self.dt)
        return dists.Normal(loc=mean, scale=std)

    def PY(self, t, xp, x):
        """
        Observation distribution: R_t | V_{t-1} = xp, V_t = x.
        
        Uses Euler discretization:
        R_t = (r - 0.5*V_{t-1})Δt + √(V_{t-1}Δt) ε_t
        
        Note: We use V_{t-1} (xp) because R_t = ln(S_t) - ln(S_{t-1}) represents
        the return over the interval [t-1, t], which depends on the variance
        at the beginning of the interval.
        
        At t=0, xp is None, so we use x (V_0) instead.
        """
        # Handle initial observation (t=0): use V_0 (x) when xp is None
        if xp is None:
            vp = np.maximum(np.asarray(x), 1e-12)  # Use V_0 at t=0
        else:
            vp = np.maximum(np.asarray(xp), 1e-12)  # Use V_{t-1} for t>0
        
        # Handle time-varying risk-free rate
        # Note: t can be 0 to len(data), but r_series has len(data) elements (indices 0 to len(data)-1)
        # For observation at time t, we use r_{t-1} (the risk-free rate from previous time step)
        # since the observation depends on V_{t-1}
        if hasattr(self.r, '__len__') and len(self.r) > 1:
            # Use r_{t-1} for observation at time t (since observation depends on V_{t-1})
            # Clamp to valid range [0, len(self.r)-1]
            if t > 0:
                r_idx = min(t - 1, len(self.r) - 1)
            else:
                r_idx = 0  # At t=0, use first risk-free rate
            r_t = self.r[r_idx]
        else:
            r_t = self.r
        
        mean = (r_t - 0.5 * vp) * self.dt
        std = np.sqrt(vp * self.dt)
        return dists.Normal(loc=mean, scale=std)
    
    def proposal0(self, data):
        """
        Initial proposal distribution for guided filter.
        
        For the guided filter, this is the optimal proposal at t=0.
        Since we observe y_0, we can use it to inform the initial proposal.
        """
        # For simplicity, use the prior (can be improved with observation y_0)
        return self.PX0()
    
    def proposal(self, t, xp, data):
        """
        Optimal proposal distribution for guided filter: p(V_t | V_{t-1} = xp, Y_t = y_t).
        
        This combines the transition density p(V_t | V_{t-1}) and the 
        observation density p(Y_t | V_{t-1}) to get a better proposal.
        
        For Heston model with observation R_t:
        - Transition: V_t ~ N(μ_trans, σ_trans²) where:
          μ_trans = V_{t-1} + κ(θ - V_{t-1})Δt
          σ_trans = σ√(V_{t-1}Δt)
        - Observation: R_t | V_{t-1} ~ N(μ_obs, σ_obs²) where:
          μ_obs = (r - 0.5*V_{t-1})Δt
          σ_obs = √(V_{t-1}Δt)
        
        The optimal proposal combines both using Bayes' rule.
        However, since the observation depends on V_{t-1} (not V_t),
        we use an approximation: use the observation to refine the transition.
        """
        vp = np.maximum(np.asarray(xp), 1e-12)
        
        # Transition parameters
        mu_trans = vp + self.kappa * (self.theta - vp) * self.dt
        sigma_trans = self.sigma * np.sqrt(vp * self.dt)
        
        # Observation parameters (using V_{t-1})
        # Note: t can be 0 to len(data), but r_series has len(data) elements (indices 0 to len(data)-1)
        # For observation at time t, we use r_{t-1} (the risk-free rate from previous time step)
        if hasattr(self.r, '__len__') and len(self.r) > 1:
            # Use r_{t-1} for observation at time t (since observation depends on V_{t-1})
            # Clamp to valid range [0, len(self.r)-1]
            if t > 0:
                r_idx = min(t - 1, len(self.r) - 1)
            else:
                r_idx = 0  # At t=0, use first risk-free rate
            r_t = self.r[r_idx]
        else:
            r_t = self.r
        
        mu_obs = (r_t - 0.5 * vp) * self.dt
        sigma_obs = np.sqrt(vp * self.dt)
        
        # Optimal proposal: combine transition and observation
        # Since observation depends on V_{t-1}, we approximate by:
        # Using the observation to adjust the transition mean
        # This is a simplified optimal proposal
        if t < len(data):
            y_t = data[t]
            # Weighted combination (simplified - full optimal would require more computation)
            # Use observation to adjust proposal mean
            innovation = y_t - mu_obs
            # Adjust transition mean based on observation
            # This is an approximation of the true optimal proposal
            mu_proposal = mu_trans + 0.5 * innovation  # Simplified adjustment
            sigma_proposal = sigma_trans * 0.9  # Slightly reduce variance
        else:
            # If no observation, use transition
            mu_proposal = mu_trans
            sigma_proposal = sigma_trans
        
        # Ensure positive variance
        sigma_proposal = np.maximum(sigma_proposal, 1e-6)
        
        return dists.Normal(loc=mu_proposal, scale=sigma_proposal)


class HestonModel:
    """
    High-level interface for Heston model with particle filtering.
    
    This class provides a simple API for:
    - Loading market data
    - Running particle filters
    - Parameter estimation
    - QMC analysis
    - Visualization
    
    Examples
    --------
    >>> from heston import HestonModel
    >>> 
    >>> # Initialize model
    >>> hest = HestonModel(dt=1/252)
    >>> 
    >>> # Load data
    >>> hest.load_data(ticker="^GSPC", start="2007-01-01", end="2025-12-31")
    >>> 
    >>> # Run particle filter
    >>> hest.fit(n_particles=2000)
    >>> print(f"Log-likelihood: {hest.log_likelihood:.2f}")
    """
    
    def __init__(
        self,
        kappa: Optional[float] = None,
        theta: Optional[float] = None,
        sigma: Optional[float] = None,
        rho: Optional[float] = None,
        r: Optional[Union[float, np.ndarray]] = None,
        dt: float = 1/252,
        v0: Optional[float] = None,
        data: Optional[List[float]] = None,
        risk_free_rate: Optional[np.ndarray] = None
    ):
        """
        Initialize Heston model.
        
        Parameters
        ----------
        kappa : float, optional
            Mean reversion speed
        theta : float, optional
            Long-run variance
        sigma : float, optional
            Volatility of volatility
        rho : float, optional
            Correlation parameter
        r : float or array-like, optional
            Risk-free rate (constant or time-varying)
        dt : float
            Time step (default: 1/252 for daily data)
        v0 : float, optional
            Initial variance (default: 0.04)
        data : list, optional
            Pre-loaded log returns data
        risk_free_rate : array-like, optional
            Pre-loaded risk-free rate data
        """
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.r = r
        self.dt = dt
        self.v0 = v0 if v0 is not None else 0.04
        
        # Data storage
        self.data = data
        self.risk_free_rate = risk_free_rate
        self.dates = None
        self.log_returns = None
        
        # Results storage
        self._ssm = None
        self._filter_result = None
        self.log_likelihood = None
        self.filtered_variance = None
        self.filtered_volatility = None
        
        # Estimated parameters
        self.estimated_params = {}
    
    def load_data(
        self,
        ticker: str = "^GSPC",
        start: str = "2007-01-01",
        end: str = "2025-12-31",
        risk_free_ticker: str = "^IRX"
    ):
        """
        Load market data from Yahoo Finance.
        
        Parameters
        ----------
        ticker : str
            Stock ticker symbol (default: "^GSPC")
        start : str
            Start date in YYYY-MM-DD format
        end : str
            End date in YYYY-MM-DD format
        risk_free_ticker : str
            Ticker for risk-free rate (default: "^IRX")
        """
        from .utils import load_market_data
        
        self.log_returns, self.risk_free_rate, self.dates = load_market_data(
            ticker=ticker,
            start=start,
            end=end,
            risk_free_ticker=risk_free_ticker
        )
        
        # Convert to lists for particles library
        self.data = self.log_returns.values.tolist()
        
        # Set risk-free rate if not already set
        if self.r is None:
            self.r = self.risk_free_rate.values
        
        logger.info(f"Loaded {len(self.data)} observations")
    
    def fit(
        self,
        n_particles: int = 2000,
        resampling: str = 'systematic',
        collect_moments: bool = True,
        verbose: bool = True
    ):
        """
        Run bootstrap particle filter on loaded data.
        
        Parameters
        ----------
        n_particles : int
            Number of particles (default: 2000)
        resampling : str
            Resampling scheme (default: 'systematic')
        collect_moments : bool
            Whether to collect moment statistics (default: True)
        verbose : bool
            Whether to print progress (default: True)
        
        Returns
        -------
        dict
            Dictionary with filter results including:
            - log_likelihood: Estimated log-likelihood
            - filtered_variance: Filtered variance estimates
            - filtered_volatility: Annualized volatility estimates
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if any(p is None for p in [self.kappa, self.theta, self.sigma, self.rho]):
            raise ValueError(
                "Model parameters not set. Provide kappa, theta, sigma, rho "
                "or call estimate_parameters() first."
            )
        
        # Create state-space model
        self._ssm = HestonSSM(
            kappa=self.kappa,
            theta=self.theta,
            sigma=self.sigma,
            rho=self.rho,
            r=self.r,
            dt=self.dt,
            v0=self.v0
        )
        
        # Create Feynman-Kac model
        fk = ssm.Bootstrap(ssm=self._ssm, data=self.data)
        
        # Collectors
        collectors = [Moments()] if collect_moments else []
        
        # Run particle filter
        if verbose:
            logger.info(f"Running particle filter with {n_particles} particles...")
        
        alg = particles.SMC(
            fk=fk,
            N=n_particles,
            resampling=resampling,
            collect=collectors
        )
        
        alg.run()
        
        # Store results
        self._filter_result = alg
        self.log_likelihood = alg.logLt
        
        if collect_moments:
            # Extract filtered variance
            self.filtered_variance = np.array(
                [m["mean"] for m in alg.summaries.moments]
            ).flatten()
            
            # Convert to volatility (already annualized, no need to multiply by sqrt(252))
            self.filtered_volatility = np.sqrt(self.filtered_variance)
        
        if verbose:
            logger.info(f"Estimated log-likelihood: {self.log_likelihood:.2f}")
        
        return {
            'log_likelihood': self.log_likelihood,
            'filtered_variance': self.filtered_variance,
            'filtered_volatility': self.filtered_volatility
        }
    
    def qmc(
        self,
        n_runs: int = 30,
        n_particles: int = 2000,
        qmc_type: str = 'sobol',
        compare_standard: bool = True,
        verbose: bool = True
    ) -> dict:
        """
        Compare QMC vs standard Monte Carlo for variance reduction.
        
        Parameters
        ----------
        n_runs : int
            Number of independent runs (default: 30)
        n_particles : int
            Number of particles per run (default: 2000)
        qmc_type : str
            Type of QMC sequence ('sobol' or 'halton', default: 'sobol')
        compare_standard : bool
            Whether to compare with standard MC (default: True)
        verbose : bool
            Whether to print progress (default: True)
        
        Returns
        -------
        dict
            QMC analysis results
        """
        from .qmc import run_qmc_analysis
        return run_qmc_analysis(
            model=self,
            n_runs=n_runs,
            n_particles=n_particles,
            qmc_type=qmc_type,
            compare_standard=compare_standard,
            verbose=verbose
        )
    
    def estimate_parameters(
        self,
        method: str = 'pmmh',
        n_iter: int = 10000,
        n_particles: int = 200,
        prior: Optional[dict] = None,
        **kwargs
    ) -> dict:
        """
        Estimate model parameters using various methods.
        
        Parameters
        ----------
        method : str
            Estimation method: 'pmmh' or 'simplex' (default: 'pmmh')
        n_iter : int
            Number of MCMC iterations for PMMH (default: 10000)
        n_particles : int
            Number of particles (default: 200 for PMMH, 2000 for simplex)
        prior : dict, optional
            Prior distributions for PMMH
        **kwargs
            Additional arguments passed to estimation methods
        
        Returns
        -------
        dict
            Estimation results
        """
        from .estimation import estimate_pmmh, estimate_simplex
        
        if method == 'pmmh':
            results = estimate_pmmh(
                model=self,
                n_iter=n_iter,
                n_particles=n_particles,
                prior=prior,
                **kwargs
            )
            # Update model parameters with posterior means
            self.estimated_params = results['posterior_means']
            self.kappa = results['posterior_means']['kappa']
            self.theta = results['posterior_means']['theta']
            self.sigma = results['posterior_means']['sigma']
            self.rho = results['posterior_means']['rho']
            
        elif method == 'simplex':
            results = estimate_simplex(
                model=self,
                n_particles=n_particles if n_particles >= 2000 else 2000,
                **kwargs
            )
            # Update model parameters
            self.estimated_params = results['estimated_params']
            self.kappa = results['estimated_params']['kappa']
            self.theta = results['estimated_params']['theta']
            self.sigma = results['estimated_params']['sigma']
            self.rho = results['estimated_params']['rho']
            
        elif method == 'em':
            from .estimation import estimate_em
            results = estimate_em(
                model=self,
                n_particles=n_particles if n_particles >= 2000 else 2000,
                **kwargs
            )
            # Update model parameters
            self.estimated_params = results['estimated_params']
            self.kappa = results['estimated_params']['kappa']
            self.theta = results['estimated_params']['theta']
            self.sigma = results['estimated_params']['sigma']
            self.rho = results['estimated_params']['rho']
            
        else:
            raise ValueError(f"Unknown estimation method: {method}. Use 'pmmh', 'simplex', or 'em'.")
        
        return results
    
    def get_results(self) -> dict:
        """
        Return comprehensive results dictionary.
        
        Returns
        -------
        dict
            Dictionary containing:
            - parameters: Model parameters
            - log_likelihood: Estimated log-likelihood
            - filtered_variance: Filtered variance estimates
            - filtered_volatility: Annualized volatility estimates
            - dates: Date index (if available)
        """
        results = {
            'parameters': {
                'kappa': self.kappa,
                'theta': self.theta,
                'sigma': self.sigma,
                'rho': self.rho,
                'v0': self.v0,
                'dt': self.dt
            },
            'log_likelihood': self.log_likelihood,
            'filtered_variance': self.filtered_variance,
            'filtered_volatility': self.filtered_volatility,
        }
        
        if self.dates is not None:
            results['dates'] = self.dates
        
        if self.estimated_params:
            results['estimated_parameters'] = self.estimated_params
        
        return results

