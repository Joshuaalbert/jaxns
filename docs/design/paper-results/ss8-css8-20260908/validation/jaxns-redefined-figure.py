"""Generate the revised paper's likelihood geometry and reference values."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from benchmarks.paper_reproduction.cases import (
    PAPER_CASES, SS8_COMPONENT_MEANS, SS8_COMPONENT_COVARIANCES,
    G8_LIKELIHOOD_MEAN, G8_LIKELIHOOD_COVARIANCE,
)
from benchmarks.paper_reproduction.references import component_log_evidence, component_log_likelihoods

out = Path('/largedata/albert/jaxns-ss8-css8-20260908/report')
out.mkdir(exist_ok=True)
references = {}
for case, spec in PAPER_CASES.items():
    _, truth = spec.build()
    row = {'log_Z': float(truth)}
    if case in ('spike_slab', 'curved_spike_slab8'):
        beta = .4 if case == 'curved_spike_slab8' else 0.
        values = np.array([component_log_evidence(np.array(m), np.array(c), beta) for m, c in zip(SS8_COMPONENT_MEANS, SS8_COMPONENT_COVARIANCES)])
        row.update(component_log_Z=values.tolist(), component_fractions=np.exp(values-logsumexp(values)).tolist(), beta=beta)
    references[case] = row
references['definition'] = dict(dimension=8, spike_covariance_scale=.43, rho=.031, beta=.4, geometric_volume_ratio=.43**4, normalized_spike_geometric_volume=.43**4/(1+.43**4), shear='x2=z2+beta*((z1-mu_k1)^2-Sigma_k11)', revision='ss8-css8-20260908')
(out/'references.json').write_text(json.dumps(references, indent=2)+'\n')
xx, yy = np.meshgrid(np.linspace(-6, 6, 600), np.linspace(-4, 6, 500))
points = np.column_stack([xx.ravel(), yy.ravel()])
fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
levels = [-15, -10, -7, -5, -3, -2, -1, -.5, 0]
for ax, (case, title) in zip(axes.flat, [('basic_mvn','G8: correlated Gaussian'), ('weak_curved_mvn8','CG8: curved Gaussian'), ('spike_slab','SS8: scale 0.43, correlation 0.031'), ('curved_spike_slab8','CSS8: SS8 with component-centered shear')]):
    if case in ('basic_mvn', 'weak_curved_mvn8'):
        means = np.asarray(G8_LIKELIHOOD_MEAN)[None,:2]
        covs = np.asarray(G8_LIKELIHOOD_COVARIANCE)[None,:2,:2]
    else:
        means = np.asarray(SS8_COMPONENT_MEANS)[:,:2]
        covs = np.asarray(SS8_COMPONENT_COVARIANCES)[:,:2,:2]
    beta = .4 if case in ('weak_curved_mvn8', 'curved_spike_slab8') else 0.
    density = logsumexp(component_log_likelihoods(points, means, covs, beta), axis=1).reshape(xx.shape)
    relative = density-density.max()
    cf = ax.contourf(xx, yy, relative, levels=levels, cmap='viridis', extend='min')
    ax.contour(xx, yy, relative, levels=[-10,-5,-2], colors='white', linewidths=.55, alpha=.8)
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), '--', color='tomato', linewidth=1.3)
    ax.set(title=title, xlabel='$x_1$', ylabel='$x_2$', xlim=(-6,6), ylim=(-4,6), aspect='equal')
fig.colorbar(cf, ax=axes, shrink=.8, label='Relative log likelihood (2D analogue)')
fig.savefig(out/'evidence_problems.pdf')
fig.savefig(out/'evidence_problems.png', dpi=150)
print(json.dumps(references, indent=2))
