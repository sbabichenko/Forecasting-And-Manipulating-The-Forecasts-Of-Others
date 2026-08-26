import numpy as np, jax, jax.numpy as jnp, time, sys
from spec_ch1 import *
lags=[0.0127,0.025,0.05,0.1,0.2,0.3,0.4]
def chebcoef(vals):
    N=len(vals); v=np.concatenate([vals, vals[-2:0:-1]]); c=np.real(np.fft.fft(v))/(N-1); c[0]/=2; c[N-1]/=2; return c[:N]
def solve_tied(Nt,Nth,m):
    G=Grid(Nt,Nth,m); M=Model(G); n=(Nt-1)*Nth
    def expand(z): g=z.reshape(Nt-1,Nth); return jnp.concatenate([g[:1],g],0)   # t=0 slice := first interior slice
    def F(z):
        g1,g2=expand(z[:n]),expand(z[n:])
        f1=jax.grad(lambda a: M.costs(expand(a),g2)[0])(z[:n]); f2=jax.grad(lambda b: M.costs(g1,expand(b))[1])(z[n:])
        return jnp.concatenate([f1,f2])
    Fj=jax.jit(F); jvpb=jax.jit(jax.vmap(lambda zz,v: jax.jvp(F,(zz,),(v,))[1], in_axes=(None,0)))
    z=jnp.zeros(2*n); t0=time.time()
    for it in range(20):
        f=Fj(z); nf=float(jnp.linalg.norm(f))
        if nf<1e-12: break
        E=jnp.eye(2*n); J=jnp.concatenate([jvpb(z,E[c:c+32]) for c in range(0,2*n,32)],0).T
        z=z+jnp.linalg.solve(J,-f)
    g1=expand(z[:n]); g2=expand(z[n:]); X,c1,c2=M.forward(g1,g2); J1=float(M.costs(g1,g2)[0])
    c=eval_at(G,np.asarray(c1),[0.5]*len(lags),[0.5-l for l in lags])
    g1=np.asarray(g1); a=np.argmin(abs(G.tn-0.5)); k=np.argmin(abs(G.thn-0.5))
    print(f'{Nt}x{Nth} tied: {it} its J1={J1:.7f} own@t=.5: '+' '.join('%7.3f'%v for v in c[:,1])+f' ({time.time()-t0:.0f}s)')
    print('   theta-coefs at t=%.2f (every 4th): '%G.tn[a]+' '.join('%.1e'%abs(x) for x in chebcoef(g1[a])[::4]))
    print('   t-coefs at theta=%.2f (every 2nd): '%G.thn[k]+' '.join('%.1e'%abs(x) for x in chebcoef(g1[:,k])[::2]))
    return g1
for cfg in [(16,16,12),(16,24,16),(24,32,20)]: solve_tied(*cfg)
