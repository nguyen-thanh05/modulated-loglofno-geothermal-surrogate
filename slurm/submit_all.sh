#!/bin/bash

echo "Submitting all baseline jobs (4 segments each; seed 42)..."

for config in configs/homogeneous_dataset/fno_m8x32x16_h64_homo.yml configs/heterogeneous_dataset/fno_m8x32x16_h64_hetero.yml \
              configs/homogeneous_dataset/fno_m4x16x8_h128_homo.yml configs/heterogeneous_dataset/fno_m4x16x8_h128_hetero.yml \
              configs/homogeneous_dataset/unet_homo.yml configs/heterogeneous_dataset/unet_hetero.yml \
              configs/homogeneous_dataset/uno_homo.yml configs/heterogeneous_dataset/uno_hetero.yml \
              configs/homogeneous_dataset/modulated_loglo_homo.yml configs/heterogeneous_dataset/modulated_loglo_hetero.yml; do
    name=$(basename "$config" .yml)
    segments=4

    JOB=""
    for segment in $(seq 1 "$segments"); do
        if [[ -n "$JOB" ]]; then
            JOB=$(sbatch --job-name="${name}_s42" \
                  --dependency=afterok:$JOB \
                  --export=ALL,CONFIG=$config,SEED=42 \
                  slurm/train.sh | awk '{print $4}')
        else
            JOB=$(sbatch --job-name="${name}_s42" \
                  --export=ALL,CONFIG=$config,SEED=42 \
                  slurm/train.sh | awk '{print $4}')
        fi
    done
    echo "$name chain submitted ($segments segments, last job: $JOB)"
done

echo "Done. Monitor with: squeue -u $USER"
