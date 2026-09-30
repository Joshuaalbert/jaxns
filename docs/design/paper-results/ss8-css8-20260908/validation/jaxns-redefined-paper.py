"""Render all paper experiment tables from R240 alone."""
import json
from pathlib import Path
import re
import numpy as np

out=Path('/largedata/albert/jaxns-ss8-css8-20260908/report')
summary=json.loads((out/'summary.json').read_text())['R240']
refs=json.loads((out/'references.json').read_text())
source=Path('/largedata/albert/git/jaxns-paper-pre289-reproduction/benchmarks/paper_reproduction/manuscript.tex')
paper=source.read_text()
paper=paper.replace('We find that it does not help with preventing mode death.', 'Evidence-only phantom conditioning leaves classic posterior mode weights unchanged.')
paper=paper.replace('We find that phantom conditioning can typically reduce evidence error by 30\\% to 50\\%.', 'On the four R240 benchmark problems, the leading 72 phantoms reduce point-estimate evidence RMSE by 17\\% to 52\\% without additional likelihood evaluations.')
new=r'''\paragraph{Spike--slab mixture (SS8).}
This problem uses two separated likelihood components with different covariance scales, weak correlation, and equal coefficients,
\begin{align}
    L_{\rm SS8}(x) = \sum_{k=1}^{2}\phi_8(x\mid\mu_k,\Sigma_k),
    \qquad \Sigma_1=0.43C,\quad\Sigma_2=C. \label{eq:ss8_likelihood}
\end{align}
Here $x\sim\mathcal{N}(0,I)$, $\mu_1=\mu_{\rm G}$, $\mu_2=-\mu_{\rm G}$, $C_{ii}=1$, and $C_{ij}=0.031$ for $i\ne j$.
The scale $0.43$ multiplies the covariance, not the standard deviation.
The likelihood is a sum of normalised component densities, with coefficient one for each component.
At a fixed component Mahalanobis radius, the geometric volume ratio is
\begin{align}
    V_1:V_2=|0.43C|^{1/2}:|C|^{1/2}=0.43^4:1
    =0.03418801:1. \label{eq:ss8_relative_volume}
\end{align}
The normalised geometric volumes are $3.31\%$ and $96.69\%$; these are not posterior mass fractions or constrained-prior volumes at a common likelihood threshold.
Gaussian conjugacy gives
\begin{align}
    \log Z_{\rm SS8}=\log\!\left[\sum_{k=1}^{2}\phi_8(\mu_k\mid0,I+\Sigma_k)\right]
    = -11.43438127. \label{eq:ss8_evidence}
\end{align}
The spike and slab contribute evidence fractions $0.60937688$ and $0.39062312$, respectively.
SS8 tests recovery of a geometrically small component with substantial posterior mass.

\paragraph{Curved spike--slab mixture (CSS8).}
CSS8 uses the same prior, latent component means, and covariances as SS8, and applies the CG8 shear separately to each component,
\begin{align}
    L_{\rm CSS8}(x)=\sum_{k=1}^{2}
    \phi_8\!\left(T_{0.4,k}^{-1}(x)\mid\mu_k,\Sigma_k\right), \label{eq:css8_likelihood}\\
    T_{\beta,k}(z)=\left(z_1,z_2+\beta\left[(z_1-\mu_{k1})^2-\Sigma_{k,11}\right],z_3,\ldots,z_8\right).
\end{align}
Each shear is centered on its own latent component mean and first-coordinate variance.
It has unit Jacobian and zero mean displacement of the second coordinate under that component.
The physical prior remains $\mathcal{N}(0,I)$.
The shear therefore preserves component contour volumes but changes their overlap with the prior.
Conditioning the remaining seven latent coordinates on $z_1$ gives a deterministic one-dimensional quadrature for each component integral,
\begin{align}
    \log Z_{\rm CSS8}
    =\log\!\left[\sum_{k=1}^{2}\E_{z\sim\mathcal{N}(\mu_k,\Sigma_k)}
    \phi_8(T_{0.4,k}(z)\mid0,I)\right]
    =-11.55909212. \label{eq:css8_evidence}
\end{align}
The curved spike and slab contribute evidence fractions $0.64575090$ and $0.35424910$.
Adaptive quadrature and independent Gauss--Hermite quadrature agree to better than $5\times10^{-11}$ in log evidence.
CSS8 isolates the effect of component-centered curvature on the weakly correlated SS8 geometry.

'''
a=paper.index(r'\paragraph{Spike--slab mixture (SS8).}')
b=paper.index(r'\begin{figure*}',a)
paper=paper[:a]+new+paper[b:]
oldcaption=re.search(r'\\caption\{Two-dimensional analogues.*?\}\n',paper).group()
caption=r'''\caption{Two-dimensional analogues of the four $D=8$ known-evidence problems. Filled contours show relative log likelihood and white curves show selected likelihood levels. The red dashed circle is the one-$\sigma$ prior contour. The lower panels share the SS8 covariance scale $0.43$ and correlation $0.031$; CSS8 adds a $\beta=0.4$ shear centered separately on each component. The displayed two-dimensional densities are analogues, not posterior marginals of the eight-dimensional problems.}
'''
paper=paper.replace(oldcaption,caption)
paper=paper.replace('Every run uses the isotropic direction kernel.','Every run uses the isotropic direction kernel, execution width $80$, and slice sampling without stepping out.\nThe sampler retains the leading $72$ phantom states for evidence reduction; only classic samples are eligible seeds in the R baseline.')
paper=paper.replace('Each problem is run with 30 different random seeds, using the same set of seeds across problems to control variability from the random-number realisations.', 'All experiment tables in this paper use the R240 baseline, with 30 seeds numbered $0$--$29$ per problem.\nR retains the standard core from commit \\texttt{a18124b}; R240 uses $240$ initial roots and uniform allocation increments of $240$ (the implementation parameter is the multiplier \\texttt{delta\\_K=1}).\nThe revised SS8/CSS8 runs use frozen experiment commit \\texttt{4071041}; G8/CG8 reuse the unchanged R240 retest cohort.\nThe source manifests and per-seed records are indexed in the accompanying reproduction report.')
paper=paper.replace('Within each problem table, bold identifies the phantom prefix with the lowest reported RMSE, with ties broken in favour of the lower prefix.', 'Mode error is reported once per classic tree in Table~\\ref{tab:r240_mode_mass}, independently of the phantom prefix.\nFor CSS8, each sample is inverse-sheared separately for each component before comparing component likelihoods.\nThe reference spike mass is the component evidence fraction; hard component assignment can introduce a small overlap discrepancy.\nBias uncertainty is the across-tree SD; RMSE uncertainty is the bootstrap SE from $10{,}000$ resamples of the 30 seeds (bootstrap seed $20260908$).\nWithin each evidence table, bold identifies the smallest point-estimate RMSE, breaking ties in favour of the lower prefix; it is not a significance statement.')

# Inline all four tables so the manuscript remains self-contained.
tables=[]
for case,label in [('basic_mvn','G8'),('weak_curved_mvn8','CG8'),('spike_slab','SS8'),('curved_spike_slab8','CSS8')]:
    r=summary[case]; best=int(np.argmin(r['rmse']))
    tables.append(r'\begin{table*}[t]')
    tables.append(r'\caption{'+f'{label} R240 paired evidence accuracy over 30 trees. '+r'Prefix sizes are multiples of $D=8$; $0D$ is classic shrinkage. Bias errors are across-tree SDs and RMSE errors are bootstrap SEs. Mean likelihood calls are $('+f"{r['mean_calls']/1e6:.3f}"+r'\pm'+f"{r['sd_calls']/1e6:.3f}"+r')\times10^6$ (mean $\pm$ across-tree SD). Bold marks the lowest point-estimate RMSE.}')
    tables.extend([r'\label{tab:phantom_evidence_'+label.lower()+'}',r'\centering',r'\scriptsize',r'\begin{tabular}{lrr}',r'\toprule',r'Phantom prefix & Bias $\log Z\mathbin{\pm}{\rm SD}$ & RMSE $\log Z\mathbin{\pm}{\rm SE}$ \\',r'\midrule'])
    for i in range(10):
        prefix=f'${i}D$'+(' (classic)' if i==0 else '')
        bias=f"{r['bias'][i]:+.4f}"+r' $\pm$ '+f"{r['bias_sd'][i]:.4f}"
        rmse=f"{r['rmse'][i]:.4f}"+r' $\pm$ '+f"{r['rmse_se'][i]:.4f}"
        if i==best:
            prefix=r'$\boldsymbol{'+f'{i}D'+r'}$'+(' (classic)' if i==0 else '')
            bias=r'\textbf{'+bias+'}'; rmse=r'\textbf{'+rmse+'}'
        tables.append(prefix+' & '+bias+' & '+rmse+r' \\')
    tables.extend([r'\bottomrule',r'\end{tabular}',r'\end{table*}',''])
tables.extend([r'\begin{table*}[t]',r'\caption{R240 classic posterior spike-mass recovery over 30 trees. Each tree contributes one posterior estimate, shared by every phantom-prefix evidence analysis. RMSE uncertainty is the bootstrap SE.}',r'\label{tab:r240_mode_mass}',r'\centering',r'\begin{tabular}{lrrr}',r'\toprule',r'Problem & Reference spike fraction & Mean recovered fraction & Spike-mass RMSE $\pm$ SE \\',r'\midrule'])
for case,label in [('spike_slab','SS8'),('curved_spike_slab8','CSS8')]:
    p=summary[case]['posterior']
    tables.append(f"{label} & {p['truth']:.5f} & {p['mean']:.5f} & {p['rmse']:.4f}"+r' $\pm$ '+f"{p['rmse_se']:.4f}"+r' \\')
tables.extend([r'\bottomrule',r'\end{tabular}',r'\end{table*}',''])
a=paper.index(r'\begin{table*}',paper.index(r'\subsection{Paired protocol and metrics}'))
b=paper.index(r'\clearpage',a)
paper=paper[:a]+'\n'.join(tables)+'\n'+paper[b:]
results=[r'\subsection{Results and limitations}','', 'At the pre-specified $9D$ phantom prefix, all four R240 problems have lower point-estimate evidence RMSE than classic shrinkage.']
for case,label in [('basic_mvn','G8'),('weak_curved_mvn8','CG8'),('spike_slab','SS8'),('curved_spike_slab8','CSS8')]:
    r=summary[case]
    improvement=100*(1-r['rmse'][9]/r['rmse'][0])
    results.append(f"For {label}, RMSE changes from ${r['rmse'][0]:.4f}$ to ${r['rmse'][9]:.4f}$, a ${improvement:.1f}\\%$ reduction.")
results.extend(['', 'The revised SS8 and CSS8 each retain substantial classic posterior mass in both components across all 30 seeds.'])
for case,label in [('spike_slab','SS8'),('curved_spike_slab8','CSS8')]:
    p=summary[case]['posterior'];r=summary[case]
    results.append(f"For {label}, the mean recovered spike fraction is ${p['mean']:.5f}$ against the reference ${p['truth']:.5f}$, with spike-mass RMSE ${p['rmse']:.4f}$; the mean number of goal iterations is ${r['mean_goals']:.2f}$.")
results.extend(['The remaining mode-mass errors are unchanged by evidence-only phantom conditioning.', 'The revised problems replace the earlier SS8 and correlated CSS8 definitions; results from those retired problems are excluded from every table.', '', r'These finite-seed comparisons measure error on completed trees, not a guarantee of mode discovery. A stopping uncertainty below $0.05$ does not itself establish calibrated empirical error when exploration and mode weights vary between runs.', 'When used only for evidence conditioning, phantoms cannot repair a missing mode in the classic posterior. The broader seed-selection and allocation interventions are reported separately from the R240 paper baseline.',''])
a=paper.index(r'\subsection{Results and limitations}')
b=paper.index(r'\section{Conclusions}',a)
paper=paper[:a]+'\n'.join(results)+'\n'+paper[b:]
paper=paper.replace('Without additional likelihood effort, we show that phantoms can reduce evidence error by approximately $30\\%$ to $50\\%$ across many classes of problems.\nThe strongly correlated spike--slab ablation shows that phantom conditioning alone cannot recover structure that is absent from the classic samples.', 'Without additional likelihood effort, the $9D$ phantom prefix reduces the point-estimate evidence RMSE by approximately $17\\%$ to $52\\%$ across these four R240 problems.\nThe revised spike--slab experiments retain both modes, with posterior mode-mass error evaluated independently of phantom-prefix evidence conditioning.')
(out/'paper.tex').write_text(paper)
(out/'R240_tables.tex').write_text('\n'.join(tables)+'\n')
assert len(re.findall(r'\\begin\{table\*\}',paper))==5
assert all(stale not in paper for stale in ('0.5989','0.5600','1:16','Correlated spike--slab mixture','0.5I','mode death.'))
# Detect duplicate labels, missing references, and mismatched environments.
labels=re.findall(r'\\label\{([^}]+)\}',paper)
assert len(labels)==len(set(labels))
references=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',paper)
assert not set(references)-set(labels), set(references)-set(labels)
stack=[]
for kind,env in re.findall(r'\\(begin|end)\{([^}]+)\}',paper):
    if kind=='begin': stack.append(env)
    else: assert stack.pop()==env
assert not stack
print('Rendered and checked five R240-only tables and revised manuscript:',out/'paper.tex')
