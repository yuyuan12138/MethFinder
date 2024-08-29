#!/bin/bash

# for i in {1..20}; do
#     python -u train.py
# done

for dir in ./*; do
  # shellcheck disable=SC2164
  cd ./"${dir##*/}"
  ../kpLogo train.fasta -gapped -pc 0.01
  # shellcheck disable=SC2103
  cd ..
done
