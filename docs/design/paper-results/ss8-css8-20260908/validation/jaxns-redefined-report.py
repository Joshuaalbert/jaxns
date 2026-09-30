"""Audit the new cohort and collect all four 240-root comparisons."""
import hashlib
import json
from pathlib import Path
import numpy as np

OUT = Path('/largedata/albert/jaxns-ss8-css8-20260908')
REPORT = OUT / 'report'
SOURCES = json.loads((OUT/'SOURCES.json').read_text())
OLD = json.loads((REPORT/'G8_CG8_INPUTS.json').read_text())
CASES = {'basic_mvn':'G8', 'weak_curved_mvn8':'CG8', 'spike_slab':'SS8', 'curved_spike_slab8':'CSS8'}
indices = np.random.default_rng(20260908).integers(0,30,(10000,30))
summary = {}
audit = {'new_cells':0, 'reused_G8_CG8_cells':0, 'bootstrap_seed':20260908, 'bootstrap_resamples':10000, 'new_source_commits':{v:s['commit'] for v,s in SOURCES.items()}}
assert json.loads((OUT/'FINISHED.json').read_text())['complete']
for variant in SOURCES:
    summary[variant] = {}
    for case in CASES:
        fresh = case in ('spike_slab','curved_spike_slab8')
        root = OUT / variant if fresh else Path(OLD[variant][case]['path'])
        analyses, cores, hashes, commits = [], [], {}, set()
        for seed in range(30):
            cell = root / case / f'seed-{seed:02d}'
            core = json.loads((cell/'CORE.json').read_text())
            a = json.loads((cell/'ANALYSIS.json').read_text())
            manifest = json.loads((cell/'MANIFEST.json').read_text())
            assert core['seed'] == a['seed'] == manifest['seed'] == seed
            assert core['case'] == a['case'] == manifest['case'] == case
            assert core['expected_log_Z_uncert'] < .05
            assert len(manifest['affinity']) == 1 and manifest['direction'] == 'isotropic'
            assert manifest['root_degree'] == 240 and manifest['shell_size'] == 80
            assert a['mc_draws'] == 2048
            if fresh:
                assert manifest['case_revision'] == 'ss8-css8-20260908'
                assert manifest['source_commit'] == SOURCES[variant]['commit']
                for filename in ('cases.py', 'references.py', 'run.py', 'prefix_sweep.py'):
                    assert manifest['protocol_sha256'][filename] == SOURCES[variant]['files'][filename]
                assert (cell/'state.pkl').is_file()
                with np.load(cell/'classic_posterior.npz') as posterior:
                    weights=np.exp(posterior['log_dp'])
                    np.testing.assert_allclose(weights.sum(), 1., atol=1e-10)
                    np.testing.assert_allclose(weights[posterior['is_spike']].sum(), a['mode_mass'], atol=1e-12)
                if variant in ('C240','Cprime240'):
                    first=json.loads((OUT/'R240'/case/f'seed-{seed:02d}'/'CORE.json').read_text())['goal_progress'][0]
                    for key in ('samples','uncertainty','expected_log_Z_mean','likelihood_evaluations','root_out_degree'):
                        np.testing.assert_allclose(core['goal_progress'][0][key], first[key], rtol=1e-12)
                audit['new_cells'] += 1
            else:
                audit['reused_G8_CG8_cells'] += 1
            with np.load(cell/'evidence_draws.npz') as data:
                draws = data['log_Z']
                assert draws.shape == (2048,10) and np.isfinite(draws).all()
                np.testing.assert_allclose(draws.mean(axis=0),a['log_Z_mean'],rtol=1e-12)
                np.testing.assert_allclose(draws.std(axis=0,ddof=1),a['log_Z_uncert'],rtol=1e-12)
            for name in ('CORE.json','ANALYSIS.json','MANIFEST.json','evidence_draws.npz'):
                hashes[str(cell/name)] = hashlib.sha256((cell/name).read_bytes()).hexdigest()
            commits.add(manifest['source_commit'])
            analyses.append(a); cores.append(core)
        truths=np.array([a['truth_log_Z'] for a in analyses])
        np.testing.assert_allclose(truths, truths[0], atol=1e-10)
        errors=np.array([a['log_Z_mean'] for a in analyses])-truths[:,None]
        calls=np.array([c['likelihood_evaluations'] for c in cores])
        goals=np.array([len(c['goal_progress']) for c in cores])
        boots=np.sqrt(np.mean(errors[indices]**2,axis=1))
        rmse=np.sqrt(np.mean(errors**2,axis=0))
        r=dict(path=str(root),source_commits=sorted(commits),truth_log_Z=float(truths[0]),errors=errors.tolist(),rmse=rmse.tolist(),rmse_se=boots.std(axis=0,ddof=1).tolist(),bias=errors.mean(axis=0).tolist(),bias_sd=errors.std(axis=0,ddof=1).tolist(),calls=calls.tolist(),mean_calls=float(calls.mean()),sd_calls=float(calls.std(ddof=1)),goals=goals.tolist(),mean_goals=float(goals.mean()),median_core_seconds=float(np.median([c['run_seconds'] for c in cores])),goal_sigma=[c['expected_log_Z_uncert'] for c in cores],record_sha256=hashes)
        if fresh:
            mass=np.array([a['mode_mass'] for a in analyses]); truth=analyses[0]['mode_mass_truth']
            e=mass-truth
            r['posterior']=dict(mass_by_seed=mass.tolist(),truth=truth,mean=float(mass.mean()),rmse=float(np.sqrt(np.mean(e**2))),rmse_se=float(np.sqrt(np.mean(e[indices]**2,axis=1)).std(ddof=1)),min=float(mass.min()),max=float(mass.max()))
        summary[variant][case]=r

# Reconstruct actual concurrency and verify one unique pinned CPU per worker.
active={}; max_active=0; completed=0
for line in (OUT/'dispatch.log').read_text().splitlines():
    if not line.startswith('{'): continue
    row=json.loads(line)
    if row['event']=='start':
        assert row['cpu'] not in active
        active[row['cpu']]=row['pid']
        max_active=max(max_active,len(active))
        assert max_active <= 60
    elif row['event']=='finish':
        assert active.pop(row['cpu']) == row['pid']
        assert row['success'] and row['exit_code']==0
        completed+=1
assert not active and completed==480
audit.update(max_active_workers=max_active,completed_phases=completed)
(REPORT/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
(REPORT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# R/A/C/C′ at 240 roots: revised SS8 and CSS8','', 'SS8 uses covariance scale 0.43 and correlation 0.031. CSS8 applies β=0.4 separately around each component mean and latent variance. Thirty seeds per row; isotropic kernel, one pinned core per worker, at most 60 concurrent workers, classic expected log-evidence uncertainty <0.05. G8 and CG8 reuse their unchanged earlier 240-root measurements.','', 'A includes all phantom U points in seed selection. C uses evidence-improving allocation after the first uniform goal. C′ uses an equal weighted sum of evidence and posterior allocation gaps after the same first goal. All evidence comparisons retain the same leading 72 phantoms. Posterior mode RMSE is computed once from classic weights and hard component labels; the reference is the component evidence fraction.','', '| Problem | Variant | Classic log Z RMSE | Ph72 log Z RMSE | Mode RMSE | Mean calls (M) | Mean goals | Median core (s) |','|---|---|---:|---:|---:|---:|---:|---:|']
for case,label in CASES.items():
    for v in SOURCES:
        r=summary[v][case]; mode=f"{r['posterior']['rmse']:.4f}" if 'posterior' in r else '—'
        lines.append(f"| {label} | {v.replace('prime','′')} | {r['rmse'][0]:.4f} ± {r['rmse_se'][0]:.4f} | {r['rmse'][9]:.4f} ± {r['rmse_se'][9]:.4f} | {mode} | {r['mean_calls']/1e6:.3f} | {r['mean_goals']:.2f} | {r['median_core_seconds']:.1f} |")
lines+=['','RMSE ± bootstrap SE uses 10,000 seed resamples. Core times include compilation and concurrent host load; likelihood counts are the primary work comparison. The mode reference is a component fraction; hard-label overlap can contribute a small systematic discrepancy.','', '## Paired evidence improvement from conditioning','', 'Positive classic-minus-Ph72 RMSE means improvement. Intervals are unadjusted paired-bootstrap 95% intervals, not corrected for multiple comparisons.','', '| Problem | Variant | Classic − Ph72 RMSE [95% CI] |','|---|---|---|']
contrasts={}
for case,label in CASES.items():
    for v in SOURCES:
        r=summary[v][case]; e=np.array(r['errors']); boot=np.sqrt(np.mean(e[indices]**2,axis=1)); d=boot[:,0]-boot[:,9]
        ci=np.quantile(d,[.025,.975]); point=r['rmse'][0]-r['rmse'][9]
        contrasts[f'{v}/{case}/conditioning']=dict(difference=point,ci95=ci.tolist())
        lines.append(f'| {label} | {v} | {point:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] |')
lines+=['','## Matched policy contrasts','', 'Differences are intervention minus R; negative RMSE differences favor the intervention.','', '| Problem | Variant − R | Classic RMSE difference [95% CI] | Ph72 RMSE difference [95% CI] | Calls ratio |','|---|---|---|---|---:|']
for case,label in CASES.items():
    base=summary['R240'][case]; be=np.array(base['errors']); bboot=np.sqrt(np.mean(be[indices]**2,axis=1))
    for v in ('A240','C240','Cprime240'):
        r=summary[v][case]; e=np.array(r['errors']); boot=np.sqrt(np.mean(e[indices]**2,axis=1))-bboot
        ci=np.quantile(boot,[.025,.975],axis=0); d=np.array(r['rmse'])-base['rmse']; ratio=r['mean_calls']/base['mean_calls']
        contrasts[f'{v}-R240/{case}']=dict(rmse_difference=d.tolist(),ci95=ci.tolist(),calls_ratio=ratio)
        entries=[f'{d[i]:+.4f} [{ci[0,i]:+.4f}, {ci[1,i]:+.4f}]' for i in (0,9)]
        lines.append(f'| {label} | {v} − R240 | '+ ' | '.join(entries)+f' | {ratio:.3f} |')
(REPORT/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
(REPORT/'contrasts.json').write_text(json.dumps(contrasts,indent=2)+'\n')
print('\n'.join(lines[:25]))
