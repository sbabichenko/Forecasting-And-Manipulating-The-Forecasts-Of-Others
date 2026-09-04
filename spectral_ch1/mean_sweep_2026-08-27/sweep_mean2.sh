#!/bin/bash
cd /home/sbabichenko/.claude/jobs/fe98da7b/tmp
echo "# precision sweep at 12x16 m=12 (Dbar1(0) converged to 0.03% vs 16x24)" > mean_sweep.txt
for p in 0.1 0.3 1 2 3 5 10 30 100 300 1000 3000; do
  r=$(./spec_ch1 12 16 12 --p1 $p --p2 $p --threads 8 --out-mean mean_p$p.txt 2>/dev/null | grep -E "^J1|^mean" | tr '\n' ' ')
  echo "p=$p : $r" >> mean_sweep.txt
done
echo DONE >> mean_sweep.txt
