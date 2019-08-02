1. 
`../data/intrim/sdUnitCircle_f_1_dt_1e-3_phis.pkl`
`../data/intrim/sdUnitCircle_f_1_dt_1e-3_deltas.pkl`

```python
n_points = 100
xlim = (-2,2)
ylim = (-2,2)
sdf = sdfs.sdUnitCircle
xs = np.linspace(*xlim, n_points)
ys = np.linspace(*ylim, n_points)[::-1]
zz = sdfs.eval_sdf(xs, ys, sdf)
base = hv.Image((xs, ys, zz))

    
ls = LSEvolver(xs,ys,zz)
F = 1
dt = 1e-3
pde_class = 'hyperbolic'
collect_every = 10
maxIter = 100
threshold=1e-6
# ls.propagate(F,dt,pde_class,True)
# run propagation steps
deltas, phis = ls.run(F, dt, pde_class, threshold=1e-6, maxIter=maxIter, collect_every=collect_every)
```

2. 
`../data/intrim/sdUnitCircle_f_1_dt_1e-2_phis.pkl`
`../data/intrim/sdUnitCircle_f_1_dt_1e-2_deltas.pkl`