#!/bin/bash

echo $1;
ID=$(sbatch --parsable $1)
shift 
for script in "$@"; do
  echo $script;
  ID=$(sbatch --parsable --dependency=afterok:$ID $script)
done

