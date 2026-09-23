## Folder: `redditRun_Sep26`

**Annotation guideline:** `annotationGuide.txt`

> Note: Files over 100 MB (raw Reddit data and large result CSVs) are excluded from this repo via `.gitignore` and are stored locally.

### 1. Reddit Data from 5 Subreddits – Burnout Classification
**Folder:** `classification_redo/`

| Input File | Code File | Output File |
|---|---|---|
| `labeled_dataset_new.xlsx` | `phase1_SVM_classification.ipynb` | `master_posts_classified.csv` |

### 2. BAT Classification (UVA HPC server using Kimi K2.5 API calls)
**Folder:** `BAT_slurm/`

| Input File | Code File | Output File |
|---|---|---|
| `master_posts_classified.csv` | `phase2a_uva_kimik2_parallel_v2.py`<br>`run_bat_array.sbatch`<br>`merge_shards.py`<br>`sample_labeling_check.py` | `bat_posts_results_final_patched.csv` |
| `bat_posts_results_final_patched.csv` | `bat_results_stats.ipynb` | No output file, but see the stats |

### 3. Exit Classification
**Folder:** `EXIT_slurm/`

| Input File | Code File | Output File |
|---|---|---|
| `bat_posts_results_final_patched.csv` | `phase2b_hpc_kimik2_parallel.py`<br>`run_exit_array.sbatch`<br>`merge_shards_exit.py` | `exit_annotated_pass1_merged.csv`<br>`exit_annotated_pass1_merged_with_dates.csv` |
| `exit_annotated_pass1_merged.csv` | `checking_file.ipynb` | No output file, just looking into stats |

### 4. BAT & EXIT Validity – 2 Coders and LLM Agreement
**Folder:** `bat_validity/`

| Code File | Output File |
|---|---|
| `bat_Annotation_validityCheck.ipynb` | All variations of annotation files are there, 60 sample then 40 sample separate |

### 5. Granger Causality Analysis, NVD Dataset – Q1
**Code folder:** `nvd_analysis/`

| Input File | Code File | Output File |
|---|---|---|
| — | `nvd_analysis/q1_timeseries.ipynb` | `q1_analysis_v2/q1_monthly_series.csv` |
| `q1_analysis_v2/q1_monthly_series.csv` | `nvd_analysis/granger_q1_monthly_v3_padjusted.ipynb` | `q1_analysis_v2/q1_granger_causality_results_v3(monthly).csv` |

### 6. Granger Causality Analysis – Q2
**Code folder:** `nvd_analysis/`

| Input File | Code File | Output File |
|---|---|---|
| — | `nvd_analysis/q2_timeseries.ipynb`<br>`Q2_monthly_series.ipynb` | `q2_analysis_v2/exit_monthly_series.csv` |
| `q2_analysis_v2/exit_monthly_series.csv` | `nvd_analysis/granger_q2_v1_monthly_padjusted.ipynb` | `q2_analysis_v2/exit_granger_causality_results(monthly).csv` |

### 7. Event Analysis
**Folder:** `eventsCheck/` · **Code folder:** `event_analysis/` · **Output folder:** `event_study_v2/`

- `build_daily_burnout_series.ipynb` – Reads `bat_posts_results_with_dates_v2.csv` (post-level). Aggregates to one row per calendar day: total post volume + counts at each BAT-score threshold. Also picks up individual construct columns (EX/EMO/COG/MD).
- `build_event_study_data.ipynb` – For each of 11 named cybersecurity incidents (2018–2026), extracts a ±30 day window of daily burnout counts around the event date, so they can be pooled and averaged across events regardless of when in the study period they occurred.
- `analyze_event_study.ipynb` – Pools all 11 events together on a common "days from event" axis and asks: averaged across very different incidents and years, is there a consistent burnout response pattern around major cybersecurity events?
- `analyse_event_pool.ipynb` – Uses a Poisson model to answer: "Across all 11 events pooled together, is burnout activity higher in the 30 days after an event than in the 30 days before, controlling for day-of-week, Patch Tuesday, each event's own baseline level, and daily Reddit post volume?"

### 8. Concept: Comment Data
**Folder:** `concept/`

The data from BAT-positive posts' top-level comments. Some phases were run in Google Colab and then locally, for posts and comments, to run LLooM.

### 9. BAT vs EXIT Correlation
**Folder:** `batExitCorr/`

| Input File | Code File | Output File |
|---|---|---|
| `BAT_slurm/bat_posts_results_final_patched.csv`<br>`EXIT_slurm/exit_annotated_pass1_merged.csv` | `burnout_exit_corr.ipynb` | `construct_exit_analysis1/` |
