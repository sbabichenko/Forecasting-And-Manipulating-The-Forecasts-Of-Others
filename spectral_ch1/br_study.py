import numpy as np, jax, jax.numpy as jnp, time, sys
from spec_ch1 import *
gf=lambda t,u: -1.5*np.exp(-(t-u)/0.15)
lags=[0.0127,0.025,0.05,0.1,0.2,0.3,0.4]
print('best response of player 1 to a fixed smooth g2; calD1 own channel at t=0.5 vs lag', flush=True)
for (Nt,Nth,m) in [(16,16,12),(16,32,20),(24,40,24)]:
    G=Grid(Nt,Nth,m); M=Model(G); n=Nt*Nth; g2=jnp.asarray(gf(G.tn[:,None],G.S))
    F=jax.jit(lambda z: jax.grad(lambda a: M.costs(a,g2)[0])(z.reshape(Nt,Nth)).ravel())
    jvpb=jax.jit(jax.vmap(lambda zz,v: jax.jvp(F,(zz,),(v,))[1], in_axes=(None,0)))
    z=jnp.zeros(n); t0=time.time()
    for it in range(20):
        f=F(z); nf=float(jnp.linalg.norm(f))
        if nf<1e-12: break
        E=jnp.eye(n); J=jnp.concatenate([jvpb(z,E[c:c+32]) for c in range(0,n,32)],0).T
        z=z+jnp.linalg.solve(J,-f)
    g1=z.reshape(Nt,Nth); X,c1,c2=M.forward(g1,g2); J1=float(M.costs(g1,g2)[0])
    c=eval_at(G,np.asarray(c1),[0.5]*len(lags),[0.5-l for l in lags])
    print(f'{Nt}x{Nth}: {it} Newton its, J1={J1:.7f}  own: '+' '.join('%7.3f'%v for v in c[:,1])+f'   ({time.time()-t0:.0f}s)', flush=True)
    np.save(f'br_{Nt}_{Nth}.npy', np.asarray(g1))
