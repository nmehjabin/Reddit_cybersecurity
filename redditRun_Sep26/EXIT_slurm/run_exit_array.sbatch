#!/bin/bash
#SBATCH --job-name=exit_kimik2
#SBATCH --account=opusmaha
#SBATCH --partition=standard
#SBATCH --array=0-59%5
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=logs/exit_kimik2_%A_%a.out
#SBATCH --error=logs/exit_kimik2_%A_%a.err

# ── Edit these two together — NUM_SHARDS must match --array above (0-59 = 60 shards) ──
# Matches the working BAT config: 60 shards, concurrency=4, %5 throttle (~20 simultaneous
# requests), ~1,040 rows/hour observed throughput on this same-size corpus.
NUM_SHARDS=60
CONCURRENCY=4

module load miniforge

cd /scratch/jqc7gj/exit_slurm

mkdir -p logs

python phase2b_hpc_kimik2_parallel.py \
    --input bat_posts_results_final_patched.csv \
    --output exit_annotated \
    --num_shards ${NUM_SHARDS} \
    --shard_id ${SLURM_ARRAY_TASK_ID} \
    --concurrency ${CONCURRENCY} \
    --id-col post_id \
    --resume
