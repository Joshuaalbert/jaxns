from jaxns.examples.joint_imaging_and_calibration.build_prior import build_prior
from jax import random


def fake_vis(nant, ndir, uncert=1.):

    prior = build_prior(nant, ndir)

    sample = prior(random.uniform(random.PRNGKey(0), shape=(prior.U_ndims,)))
    y = sample['delta']
    y_obs = y + uncert*random.normal(random.PRNGKey(1), shape=y.shape)

    return sample['theta'], sample['gamma'], y, y_obs
