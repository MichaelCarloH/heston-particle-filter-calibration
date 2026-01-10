"""
Simple import tests to verify package structure.
"""

def test_import_heston():
    """Test that heston package can be imported."""
    from heston import HestonModel, HestonSSM
    assert HestonModel is not None
    assert HestonSSM is not None


def test_heston_model_init():
    """Test HestonModel initialization."""
    from heston import HestonModel
    
    hest = HestonModel(dt=1/252)
    assert hest.dt == 1/252
    assert hest.v0 == 0.04  # default value


def test_heston_ssm_init():
    """Test HestonSSM initialization."""
    from heston import HestonSSM
    
    ssm = HestonSSM(
        kappa=2.0,
        theta=0.04,
        sigma=0.3,
        rho=-0.7,
        r=0.04,
        dt=1/252,
        v0=0.04
    )
    
    assert ssm.kappa == 2.0
    assert ssm.theta == 0.04
    assert ssm.sigma == 0.3
    assert ssm.rho == -0.7


if __name__ == "__main__":
    test_import_heston()
    test_heston_model_init()
    test_heston_ssm_init()
    print("All import tests passed!")

