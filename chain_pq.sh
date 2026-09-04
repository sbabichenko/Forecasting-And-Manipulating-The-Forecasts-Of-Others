#!/bin/bash
# continuation in p_Q at the target cost, warm-starting from the previous tilt
cd /home/sbabichenko/.claude/jobs/fe98da7b/tmp
N=$1; L=$2; shift 2; prev=""
echo "# N=$N L=$L" > chain_N${N}_L${L}.txt
for pq in "$@"; do
  if [ -z "$prev" ]; then
    KB_BLAS_THREADS=4 timeout 3000 ./kb_inv $N $L 0.001 0 1,1,1 --pq $pq --gq 1 --eps-path 0.3,0.1,0.03,0.01,0.003,0.001 --uniform 201 --tol 1e-10 --threads 4 --nk --dump-z z_N${N}_L${L}_pq$pq.txt > ch_N${N}_L${L}_pq$pq.json 2>/dev/null
  else
    KB_BLAS_THREADS=4 timeout 3000 ./kb_inv $N $L 0.001 0 1,1,1 --pq $pq --gq 1 --eps-path 0.001 --uniform 201 --tol 1e-10 --threads 4 --warm z_N${N}_L${L}_pq$prev.txt --dump-z z_N${N}_L${L}_pq$pq.txt > ch_N${N}_L${L}_pq$pq.json 2>/dev/null
  fi
  python3 - <<PY >> chain_N${N}_L${L}.txt
import json,math
try:
    d=json.load(open('ch_N${N}_L${L}_pq$pq.json')); I=d['inventory']; t=d['traders'][0]
    print('pq=%-6s conv %-5s res %.0e lambda %.4f flow %.4f | VarQ %.3f q(L) %.4f profit %+.4f B0 %.2f mm_loss %.3f | implied p_Q %.3f'%('$pq',d['converged'],d['residual'],d['lambda'],t['flow'],I['varQ'],I['q_at_L'],I['profit_informed'],I['B0'],I['mm_loss'],-math.sqrt(1.0/I['B0']) if I['B0']>0 else float('nan')))
except Exception as e: print('pq=$pq failed', e)
PY
  prev=$pq
done
echo DONE >> chain_N${N}_L${L}.txt
