"""
Diagnostic: the FULL deseasonalization regression output for all four
series, instead of just the one-line R^2 summary printed inside
granger_q1_analysis.py.

This runs the exact same regression deseasonalize() uses internally:
    y = const + Tue + Wed + Thu + Fri + Sat + Sun + patch_tuesday + leftover
(Monday is the reference day every other coefficient is compared against)

But here it prints the COMPLETE statsmodels OLS table -- every single
day's coefficient, its standard error, t-statistic, and its own p-value --
so you can see exactly which days are significantly different from
Monday for EACH series, and compare burnout's weekly pattern against
CVE's weekly pattern directly, before any causality testing happens.
"""

import pandas as pd
import statsmodels.api as sm
import os

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
OUT_DIR = "/Users/nadia/Desktop/redditRun_june/q1_analysis/"
DAILY_CSV = os.path.join(OUT_DIR, "q1_daily_series.csv")

VALUE_COLS = ["n_bat_ge1", "n_bat_ge2", "n_cve_run1", "n_cve_run2"]

DAY_NAMES = {1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}  # 0=Monday is the dropped baseline


def is_patch_tuesday(date):
    return 1 if (date.weekday() == 1 and 8 <= date.day <= 14) else 0


def main():
    if not os.path.exists(DAILY_CSV):
        print(f"Could not find {DAILY_CSV}. Run build_q1_timeseries.py first.")
        return

    daily = pd.read_csv(DAILY_CSV)
    dates = pd.to_datetime(daily["date"])
    dow = dates.dt.dayofweek
    patch_tues = dates.apply(is_patch_tuesday)

    dow_dummies = pd.get_dummies(dow, prefix="dow", drop_first=True)
    dow_dummies = dow_dummies.rename(columns={f"dow_{k}": v for k, v in DAY_NAMES.items()})

    X = pd.concat([dow_dummies, patch_tues.rename("patch_tuesday")], axis=1)
    X = sm.add_constant(X).astype(float)

    print("Reference day (the baseline everything else is compared against): Monday\n")

    for col in VALUE_COLS:
        print(f"\n{'=' * 75}")
        print(f"FULL DESEASONALIZATION REGRESSION:  {col}")
        print("=" * 75)
        y = daily[col].astype(float)
        model = sm.OLS(y, X).fit()
        print(model.summary())
        print(f"\n  Plain-language read: any row above with P>|t| < 0.05 is a day that's")
        print(f"  STATISTICALLY DIFFERENT from a typical Monday for {col}. Compare which")
        print(f"  days light up here against which days light up for the OTHER series below.")


if __name__ == "__main__":
    main()