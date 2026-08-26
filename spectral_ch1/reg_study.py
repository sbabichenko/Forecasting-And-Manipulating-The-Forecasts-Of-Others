import numpy as np, jax, jax.numpy as jnp, time
from spec_ch1 import *
lags=[0.0127,0.025,0.05,0.1,0.2,0.3,0.4]
def chebcoef(vals):
    N=len(vals); v=np.concatenate([vals, vals[-2:0:-1]]); c=np.real(np.fft.fft(v))/(N-1); c[0]/=2; c[N-1]/=2; return c[:N]
def diffmat(x):   # Chebyshev differentiation matrix on arbitrary nodes (barycentric)
    N=len(x); w=bary_w(N); D=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i!=j: D[i,j]=w[j]/w[i]/(x[i]-x[j])
        D[i,i]=-D[i].sum()
    return D
def run(Nt,Nth,m,lam,save=False):
    G=Grid(Nt,Nth,m); M=Model(G); n=(Nt-1)*Nth
    Dt=jnp.asarray(diffmat(G.tn)); Dth=jnp.asarray(diffmat(G.thn)); Wj=G.Wj
    def expand(z): g=z.reshape(Nt-1,Nth); return jnp.concatenate([g[:1],g],0)
    def pen(g):   # lam * int (d^2 g/dtheta^2)^2 + (d^2 g/dt^2)^2 over the triangle
        return lam*(jnp.sum(Wj*(g@Dth.T@Dth.T)**2)+jnp.sum(Wj*(Dt@Dt@g)**2))
    def F(z):
        g1,g2=expand(z[:n]),expand(z[n:])
        f1=jax.grad(lambda a: M.costs(expand(a),g2)[0]+pen(expand(a)))(z[:n]); f2=jax.grad(lambda b: M.costs(g1,expand(b))[1]+pen(expand(b)))(z[n:])
        return jnp.concatenate([f1,f2])
    Fj=jax.jit(F); jvpb=jax.jit(jax.vmap(lambda zz,v: jax.jvp(F,(zz,),(v,))[1], in_axes=(None,0)))
    z=jnp.zeros(2*n); t0=time.time(); J=None
    for it in range(20):
        f=Fj(z); nf=float(jnp.linalg.norm(f))
        if nf<1e-12: break
        E=jnp.eye(2*n); J=jnp.concatenate([jvpb(z,E[c:c+32]) for c in range(0,2*n,32)],0).T
        z=z+jnp.linalg.solve(J,-f)
    g1=expand(z[:n]); g2=expand(z[n:]); X,c1,c2=M.forward(g1,g2); J1=float(M.costs(g1,g2)[0])
    c=eval_at(G,np.asarray(c1),[0.5]*len(lags),[0.5-l for l in lags]); g1=np.asarray(g1); a=np.argmin(abs(G.tn-0.5)); k=np.argmin(abs(G.thn-0.5))
    print(f'{Nt}x{Nth} lam={lam:.0e}: J1={J1:.7f} own@t=.5: '+' '.join('%7.3f'%v for v in c[:,1])+f' ({time.time()-t0:.0f}s)', flush=True)
    print('   theta-coefs (every 4th): '+' '.join('%.1e'%abs(x) for x in chebcoef(g1[a])[::4])+' | t-coefs (every 2nd): '+' '.join('%.1e'%abs(x) for x in chebcoef(g1[:,k])[::2]), flush=True)
    if save:
        Jn=np.asarray(J); u,s,vt=np.linalg.svd(Jn); print('   sv max %.1e min %.1e; smallest right-singular vectors (player-1 block mass): '%(s[0],s[-1]))
        for kk in [-1,-2,-3,-4,-6]:
            v=vt[kk][:n].reshape(Nt-1,Nth)**2; tot=v.sum()+1e-300; th=v.sum(0)/tot; tt=v.sum(1)/tot
            print('     sv %.1e: theta<0.1 %.2f mid %.2f >0.9 %.2f | t<0.15 %.2f  t>0.85 %.2f'%(s[kk],th[G.thn<0.1].sum(),th[(G.thn>=0.1)&(G.thn<=0.9)].sum(),th[G.thn>0.9].sum(),tt[G.tn[1:]<0.15].sum(),tt[G.tn[1:]>0.85].sum()), flush=True)
run(16,24,16,0.0,save=True)
for lam in [1e-7,1e-5]:
    for cfg in [(16,24,16),(24,32,20)]: run(*cfg,lam)
