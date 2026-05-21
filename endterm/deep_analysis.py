"""
Step 4: Deep Analysis - Find the key pattern in "good" predictions
Focus on CLTV ranges and feature combinations
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STEP 4: DEEP ANALYSIS - Finding Key Pattern")
print("="*80)

# Load indices
good_indices_train = np.load('good_indices_train.npy')
bad_indices_train = np.load('bad_indices_train.npy')

df_full = pd.read_csv('train_BRCpofr.csv')
target = 'cltv'

# Get train indices
np.random.seed(42)
_, indices_test = train_test_split(np.arange(len(df_full)), test_size=0.3, random_state=42)
indices_train_all = np.setdiff1d(np.arange(len(df_full)), indices_test)

df_train_full = df_full.iloc[indices_train_all].reset_index(drop=True)

good_mask = np.isin(indices_train_all, good_indices_train)
df_good = df_train_full[good_mask].copy()
df_bad = df_train_full[~good_mask].copy()

print(f"\nGood predictions: {len(df_good)} samples")
print(f"Bad predictions: {len(df_bad)} samples")

# === KEY INSIGHT: CLTV Range ===
print(f"\n" + "="*80)
print("KEY INSIGHT: CLTV VALUE RANGES")
print("="*80)

print(f"\nGood predictions CLTV distribution:")
print(f"  Mean: {df_good['cltv'].mean():,.0f}")
print(f"  Median: {df_good['cltv'].median():,.0f}")
print(f"  Std: {df_good['cltv'].std():,.0f}")
print(f"  Min: {df_good['cltv'].min():,.0f}")
print(f"  Max: {df_good['cltv'].max():,.0f}")
print(f"  Percentiles:")
for p in [10, 25, 50, 75, 90]:
    val = df_good['cltv'].quantile(p/100)
    print(f"    {p}th: {val:,.0f}")

print(f"\nBad predictions CLTV distribution:")
print(f"  Mean: {df_bad['cltv'].mean():,.0f}")
print(f"  Median: {df_bad['cltv'].median():,.0f}")
print(f"  Std: {df_bad['cltv'].std():,.0f}")
print(f"  Min: {df_bad['cltv'].min():,.0f}")
print(f"  Max: {df_bad['cltv'].max():,.0f}")
print(f"  Percentiles:")
for p in [10, 25, 50, 75, 90]:
    val = df_bad['cltv'].quantile(p/100)
    print(f"    {p}th: {val:,.0f}")

# === Find optimal CLTV range ===
print(f"\n" + "="*80)
print("FINDING OPTIMAL CLTV RANGE")
print("="*80)

# Check what % of good is in different ranges
ranges = [
    (0, 50000, "0-50k"),
    (50000, 75000, "50-75k"),
    (75000, 100000, "75-100k"),
    (100000, 150000, "100-150k"),
    (150000, 1000000, "150k+")
]

print(f"\nCLTV distribution comparison:")
print(f"  {'Range':<15} {'% Good':<12} {'% Bad':<12} {'Good/Bad Ratio':<15}")
print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*15}")

for lower, upper, label in ranges:
    good_in_range = ((df_good['cltv'] >= lower) & (df_good['cltv'] < upper)).sum()
    bad_in_range = ((df_bad['cltv'] >= lower) & (df_bad['cltv'] < upper)).sum()
    
    pct_good = (good_in_range / len(df_good)) * 100
    pct_bad = (bad_in_range / len(df_bad)) * 100
    ratio = (good_in_range / len(df_good)) / (bad_in_range / len(df_bad)) if bad_in_range > 0 else 0
    
    print(f"  {label:<15} {pct_good:<12.1f} {pct_bad:<12.1f} {ratio:<15.2f}x")

# === Feature combinations in good predictions ===
print(f"\n" + "="*80)
print("FEATURE COMBINATIONS ANALYSIS")
print("="*80)

# Single policy holders (most common in good)
print(f"\nSingle Policy Holders (num_policies='1'):")
good_single = df_good[df_good['num_policies'] == '1']
bad_single = df_bad[df_bad['num_policies'] == '1']
print(f"  Good: {len(good_single)/len(df_good)*100:.1f}% (of good)")
print(f"  Bad: {len(bad_single)/len(df_bad)*100:.1f}% (of bad)")
print(f"  Good CLTV: mean={good_single['cltv'].mean():,.0f}")
print(f"  Bad CLTV: mean={bad_single['cltv'].mean():,.0f}")

# Rural area
print(f"\nRural Area:")
good_rural = df_good[df_good['area'] == 'Rural']
bad_rural = df_bad[df_bad['area'] == 'Rural']
print(f"  Good: {len(good_rural)/len(df_good)*100:.1f}% (of good)")
print(f"  Bad: {len(bad_rural)/len(df_bad)*100:.1f}% (of bad)")
print(f"  Good CLTV: mean={good_rural['cltv'].mean():,.0f}")
print(f"  Bad CLTV: mean={bad_rural['cltv'].mean():,.0f}")

# Lower income
print(f"\nIncome <= 5L-10L:")
good_low_income = df_good[df_good['income'].isin(['2L-5L', '5L-10L'])]
bad_low_income = df_bad[df_bad['income'].isin(['2L-5L', '5L-10L'])]
print(f"  Good: {len(good_low_income)/len(df_good)*100:.1f}% (of good)")
print(f"  Bad: {len(bad_low_income)/len(df_bad)*100:.1f}% (of bad)")
print(f"  Good CLTV: mean={good_low_income['cltv'].mean():,.0f}")
print(f"  Bad CLTV: mean={bad_low_income['cltv'].mean():,.0f}")

# === COMPOSITE FILTER: Find subset that model predicts best ===
print(f"\n" + "="*80)
print("COMPOSITE FILTER: Identifying Sweet Spot")
print("="*80)

# Create composite filter: Single policy + Lower CLTV
df_good_filtered = df_good[
    (df_good['num_policies'] == '1') & 
    (df_good['cltv'] <= 100000)
]

df_bad_filtered = df_bad[
    (df_bad['num_policies'] == '1') & 
    (df_bad['cltv'] <= 100000)
]

print(f"\nFilter: num_policies='1' AND cltv <= 100k")
print(f"  Good samples in this range: {len(df_good_filtered)} ({len(df_good_filtered)/len(df_good)*100:.1f}%)")
print(f"  Bad samples in this range: {len(df_bad_filtered)} ({len(df_bad_filtered)/len(df_bad)*100:.1f}%)")
print(f"  Ratio (Good/Bad): {len(df_good_filtered)/len(df_bad_filtered):.3f}")

# More aggressive filter
df_good_filtered2 = df_good[
    (df_good['num_policies'] == '1') & 
    (df_good['cltv'] <= 80000) &
    (df_good['claim_amount'] < 5000)
]

df_bad_filtered2 = df_bad[
    (df_bad['num_policies'] == '1') & 
    (df_bad['cltv'] <= 80000) &
    (df_bad['claim_amount'] < 5000)
]

print(f"\nMore aggressive filter: num_policies='1' AND cltv <= 80k AND claim < 5k")
print(f"  Good samples: {len(df_good_filtered2)} ({len(df_good_filtered2)/len(df_good)*100:.1f}%)")
print(f"  Bad samples: {len(df_bad_filtered2)} ({len(df_bad_filtered2)/len(df_bad)*100:.1f}%)")
print(f"  Ratio (Good/Bad): {len(df_good_filtered2)/len(df_bad_filtered2):.3f}")

# === STRATEGY 4: Train on lower CLTV range ===
print(f"\n" + "="*80)
print("STRATEGY 4: Train on Lower CLTV Values (High Predictability Zone)")
print("="*80)

# Filter original dataset to keep only samples with CLTV <= 100k
df_full_filtered = df_train_full[
    (df_train_full['num_policies'] == '1') & 
    (df_train_full['cltv'] <= 100000)
]

print(f"\nFiltered training set: {len(df_full_filtered)} samples")
print(f"  CLTV range: [{df_full_filtered['cltv'].min():,.0f}, {df_full_filtered['cltv'].max():,.0f}]")
print(f"  CLTV mean: {df_full_filtered['cltv'].mean():,.0f}")

# Also create stratified version: All good + some bad from this range
df_train_good_all = df_train_full[df_train_full.index.isin(
    np.where(good_mask)[0]
)]
df_train_bad_range = df_train_full[
    (df_train_full['num_policies'] == '1') & 
    (df_train_full['cltv'] <= 100000) &
    (df_train_full.index.isin(np.where(~good_mask)[0]))
]

df_train_strategy4 = pd.concat([df_train_good_all, df_train_bad_range])
df_train_strategy4 = df_train_strategy4.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nStrategy 4 dataset (Good + Bad from predictable range):")
print(f"  Total: {len(df_train_strategy4)} samples")
print(f"  Good: {len(df_train_good_all)}")
print(f"  Bad (from low CLTV range): {len(df_train_bad_range)}")

# Save this dataset
df_train_strategy4.to_csv('train_STRATEGY4_low_cltv_range.csv', index=False)
print(f"  ✓ Saved to train_STRATEGY4_low_cltv_range.csv")

print(f"\n" + "="*80)
print("✓ STEP 4 COMPLETE - Key patterns identified!")
print("="*80)

print(f"\nKey Findings:")
print(f"  1. Good predictions: Lower CLTV (mean={df_good['cltv'].mean():,.0f} vs {df_bad['cltv'].mean():,.0f})")
print(f"  2. Good predictions: More single-policy holders (56% vs 28%)")
print(f"  3. Sweet spot: CLTV <= 100k with single policy")
print(f"  4. Strategy: Train on samples with lower predictability zone!")
