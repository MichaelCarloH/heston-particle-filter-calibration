"""
Quasi-Monte Carlo (QMC) analysis for variance reduction.
"""

import numpy as np
from typing import Dict, Optional
import scipy.stats
import logging
from .model import HestonModel, HestonSSM
import particles
from particles import state_space_models as ssm
from particles.collectors import Moments

logger = logging.getLogger(__name__)


def run_qmc_analysis(
    model: HestonModel,
    n_runs: int = 30,
    n_particles: int = 2000,
    qmc_type: str = 'sobol',
    compare_standard: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Compare QMC vs standard Monte Carlo for variance reduction.
    
    Parameters
    ----------
    model : HestonModel
        Initialized Heston model with data loaded
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
        Dictionary containing:
        - standard: Mean and variance of standard MC results
        - qmc: Mean and variance of QMC results
        - variance_reduction_pct: Percentage variance reduction
        - t_test: Results of t-test comparing means
        - f_test: Results of F-test comparing variances
    """
    if model.data is None:
        raise ValueError("No data loaded. Call load_data() first.")
    
    if any(p is None for p in [model.kappa, model.theta, model.sigma, model.rho]):
        raise ValueError("Model parameters not set.")
    
    standard_results = []
    qmc_results = []
    
    # Create SSM
    ssm_model = HestonSSM(
        kappa=model.kappa,
        theta=model.theta,
        sigma=model.sigma,
        rho=model.rho,
        r=model.r,
        dt=model.dt,
        v0=model.v0
    )
    
    if verbose:
        logger.info(f"Running {n_runs} independent filter runs...")
        logger.info(f"  Standard MC: {n_runs if compare_standard else 0} runs")
        logger.info(f"  QMC ({qmc_type}): {n_runs} runs")
    
    # Use particles.multiSMC to run both standard MC and QMC
    # This is the proper way to use QMC in the particles library
    fk = ssm.Bootstrap(ssm=ssm_model, data=model.data)
    
    if compare_standard:
        # Run both standard MC and QMC using multiSMC
        # qmc parameter: {'SMC': True, 'SQMC': True} runs both algorithms
        if verbose:
            logger.info("Running multiSMC with both SMC and SQMC...")
        
        all_results = particles.multiSMC(
            fk=fk,
            N=n_particles,
            resampling='systematic',
            collect=[Moments()],
            nruns=n_runs,
            qmc={'SMC': True, 'SQMC': True},  # Run both SMC and SQMC
            nprocs=0  # Use all available cores
        )
        
        # Extract results based on qmc field
        for result in all_results:
            if result.get('qmc') == 'SMC':
                standard_results.append(result['output'].logLt)
            elif result.get('qmc') == 'SQMC':
                qmc_results.append(result['output'].logLt)
    else:
        # Only run QMC (SQMC)
        if verbose:
            logger.info("Running SQMC only...")
        qmc_multi_results = particles.multiSMC(
            fk=fk,
            N=n_particles,
            resampling='systematic',
            collect=[Moments()],
            nruns=n_runs,
            qmc={'SMC': False, 'SQMC': True},  # Run only SQMC
            nprocs=0
        )
        
        # Extract QMC results
        for result in qmc_multi_results:
            if result.get('qmc') == 'SQMC':
                qmc_results.append(result['output'].logLt)
    
    # Statistical analysis
    results = {}
    
    if compare_standard and len(standard_results) > 0:
        standard_mean = np.mean(standard_results)
        standard_var = np.var(standard_results, ddof=1)
        results['standard'] = {
            'mean': standard_mean,
            'var': standard_var,
            'std': np.sqrt(standard_var),
            'results': standard_results
        }
    
    qmc_mean = np.mean(qmc_results)
    qmc_var = np.var(qmc_results, ddof=1)
    results['qmc'] = {
        'mean': qmc_mean,
        'var': qmc_var,
        'std': np.sqrt(qmc_var),
        'results': qmc_results
    }
    
    # Variance reduction
    if compare_standard and len(standard_results) > 0:
        variance_reduction = (standard_var - qmc_var) / standard_var * 100
        results['variance_reduction_pct'] = variance_reduction
        
        # Statistical tests
        # t-test for means
        t_stat, t_pvalue = scipy.stats.ttest_ind(standard_results, qmc_results)
        results['t_test'] = {
            'statistic': t_stat,
            'pvalue': t_pvalue,
            'significant': t_pvalue < 0.05
        }
        
        # F-test for variances
        f_stat = standard_var / qmc_var if qmc_var > 0 else np.inf
        f_pvalue = 2 * min(
            scipy.stats.f.cdf(f_stat, len(standard_results)-1, len(qmc_results)-1),
            1 - scipy.stats.f.cdf(f_stat, len(standard_results)-1, len(qmc_results)-1)
        )
        results['f_test'] = {
            'statistic': f_stat,
            'pvalue': f_pvalue,
            'significant': f_pvalue < 0.05
        }
        
        if verbose:
            logger.info("\n=== QMC Analysis Results ===")
            logger.info(f"Standard MC:")
            logger.info(f"  Mean: {standard_mean:.4f}")
            logger.info(f"  Std:  {np.sqrt(standard_var):.4f}")
            logger.info(f"QMC ({qmc_type}):")
            logger.info(f"  Mean: {qmc_mean:.4f}")
            logger.info(f"  Std:  {np.sqrt(qmc_var):.4f}")
            logger.info(f"Variance reduction: {variance_reduction:.2f}%")
            logger.info(f"t-test p-value: {t_pvalue:.4f}")
            logger.info(f"F-test p-value: {f_pvalue:.4f}")
    else:
        if verbose:
            logger.info("\n=== QMC Results ===")
            logger.info(f"Mean: {qmc_mean:.4f}")
            logger.info(f"Std:  {np.sqrt(qmc_var):.4f}")
    
    return results

