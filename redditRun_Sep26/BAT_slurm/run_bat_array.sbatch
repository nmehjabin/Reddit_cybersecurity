#!/bin/bash
#SBATCH --job-name=bat_kimik2
#SBATCH --account=opusmaha
#SBATCH --partition=standard
#SBATCH --array=0-59%5        # 60 shards total, but only 5 running AT ONCE (throttle)
#SBATCH --time=12:00:00       # per-shard budget — see math below, ~9.1h expected + margin
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/bat_kimik2_%A_%a.out
#SBATCH --error=logs/bat_kimik2_%A_%a.err

# ── WHY THE THROTTLE (%5) ────────────────────────────────────────────────────
# The full 20-shard x concurrency=8 run (160 simultaneous API calls) produced
# widespread {"detail":"'messages'"} errors — including on posts that passed
# cleanly in single-shard tests at max 8 concurrent requests. That's strong
# evidence the shared UVA GenAI endpoint has a concurrency ceiling somewhere
# between 8 and 160, and request bodies get corrupted/dropped above it.
#
# This config caps TOTAL concurrent requests at 5 shards x 4 concurrency = 20,
# comfortably above the proven-safe 8 but far below the 160 that broke.
# Slurm's %5 syntax means: 60 array tasks are queued, but only 5 run at a
# time — the rest wait and start automatically as running ones finish.
#
# Total job completion time at 20 concurrent, ~54s/call: ~144,652 x 54 / 20 /
# 3600 ≈ 109 hours (~4.5 days) wall-clock, regardless of shard count, since
# total throughput is capped by total concurrency, not shard granularity.
#
# ── IF YOU WANT TO GO FASTER LATER ──────────────────────────────────────────
# Once this runs clean for a while with zero 'messages' errors, you can
# cautiously raise the throttle (e.g. %5 -> %8) on a NEW submission to find
# the real ceiling incrementally, rather than jumping back to 160. Don't
# modify a running array job's throttle — cancel and resubmit with the new
# value instead.

NUM_SHARDS=60
CONCURRENCY=4

mkdir -p logs

module load miniforge
# conda activate <your_env_name_if_any>

cd /scratch/jqc7gj/BAT_slurm/

python -u phase2a_uva_kimik2_parallel.py \
    --input master_posts_burnout_only.csv \
    --output bat_posts_results.csv \
    --num_shards $NUM_SHARDS \
    --shard_id $SLURM_ARRAY_TASK_ID \
    --concurrency $CONCURRENCY \
    --batch_size 100 \
    --resume

# Submit with:      sbatch run_bat_array.sbatch
# Check status:     squeue -u $USER
# Watch one shard:  tail -f logs/bat_kimik2_<jobid>_0.out
# Health check:     grep -c "400 Bad Request" logs/bat_kimik2_<jobid>_*.out
# If a shard fails or times out, resubmit just that index:
#   sbatch --array=3 run_bat_array.sbatch
# (the --resume flag means it picks up where that shard left off)
