#!/usr/bin/env bash
set -euo pipefail

# Lista de Ns a simular
NS=(10 20 30 40 50 60 70 80 90 100 200 300 400 500)

for N in "${NS[@]}"; do
  for rep in 1 2 3; do
    name="test${N}_600_${rep}"
    echo ">>> Running N=${N} rep=${rep} -> ${name}"
    java -cp target/classes \
      -DN="${N}" \
      -Dt_f=600 \
      -Dname="${name}" \
      ar.edu.itba.sds.tp5.simulations.Engine
  done
done
