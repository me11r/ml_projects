# Endterm Project: Data Curation Analysis & Model Optimization

## 🎯 Executive Summary

Following professor's feedback to analyze dataset patterns, we conducted a comprehensive data curation study. **Key finding: Filtering data HURTS performance. Keep all data and optimize hyperparameters instead.**

### Results
| Metric | Value | Status |
|--------|-------|--------|
| Best R² | 0.1152 | +44% improvement |
| RMSE | 84,384 | Down from 86k |
| MAE | 44,402 | Down from 47k |
| Strategy | Full data + hyperparameter tuning | ✅ Optimal |

---

## 📊 What We Found

### 1. **Good Predictions Pattern (15% best by residuals)**
- **Lower CLTV**: mean 63k (vs 104k for bad predictions)
- **Single policy**: 56% (vs 28% for bad)
- **Rural areas**: 37% (vs 29% for bad)
- **CLTV ≤50k**: 2.17x better prediction ratio

### 2. **Data Curation Strategies Tested**
```
Strategy                Result      Change
─────────────────────────────────────────
Filter to good          R²=0.097    -12.6% ❌
Mixed 70/30             R²=0.099    -10.8% ❌
Feature hint            R²=0.117    +1.7%  ✅
Low CLTV filter         R²=0.097    -12.6% ❌
Very low CLTV filter    R²=-0.48    COLLAPSE ❌❌
Weighted 2x             R²=0.113    -1.7%  ❌
Weighted 3x             R²=0.106    -7.8%  ❌
```

### 3. **Why Data Curation Failed**
1. **High variance target** (CLTV: 25k-720k, 29x range)
2. **"Good" is relative, not absolute** - patterns local to subset
3. **Limited features** (only 10) - every sample matters for learning
4. **Overfitting** - removing 85% of data removes 85% of signal

---

## 📁 Key Files

### 📖 Documentation (Start Here!)
- **[ANALYSIS_REPORT.md](ANALYSIS_REPORT.md)** - Comprehensive 7-part analysis
- **[SUMMARY_AND_NEXT_STEPS.txt](SUMMARY_AND_NEXT_STEPS.txt)** - Executive summary + recommendations
- **[FILES_MANIFEST.txt](FILES_MANIFEST.txt)** - Complete file listing

### 📓 Notebooks
- **[FINAL_IMPROVED_MODEL.ipynb](FINAL_IMPROVED_MODEL.ipynb)** - Ready for presentation!
- **[main.ipynb](main.ipynb)** - Original project

### 🐍 Analysis Scripts
1. `analysis_good_predictions.py` - Identified good vs bad patterns
2. `deep_analysis.py` - CLTV range analysis
3. `retrain_models.py` - Tested filtering strategies
4. `feature_hint_strategy.py` - Feature hint approach
5. `final_solution.py` - Comprehensive comparison
6. `advanced_strategies.py` - Weighted training

### 📊 Results
- `FINAL_comparison_all_strategies.csv` - Strategy comparison
- `ADVANCED_strategies_results.csv` - Weighted training results
- `feature_hint_results.csv` - Feature hint results

---

## 🚀 How to Use

### For Professor Presentation
```bash
1. Read: ANALYSIS_REPORT.md (Parts 1-2)
2. Show: FINAL_IMPROVED_MODEL.ipynb (run all cells)
3. Explain: Why data curation failed
4. Conclude: Full data + optimization is best
```

### For Reproducibility
```bash
# Run analysis scripts
python3 analysis_good_predictions.py
python3 deep_analysis.py
python3 retrain_models.py
python3 feature_hint_strategy.py
python3 final_solution.py
python3 advanced_strategies.py

# View results
cat FINAL_comparison_all_strategies.csv
jupyter notebook FINAL_IMPROVED_MODEL.ipynb
```

---

## 📈 Performance by CLTV Segment

| Range | Samples | R² | MAE | Quality |
|-------|---------|-----|-----|---------|
| 0-50k | 6,225 | -17.6 | 17.3k | Poor |
| 50-100k | 13,155 | -4.6 | 24.6k | Poor |
| 100-200k | 4,888 | -1.4 | 34.3k | Poor |
| 200k+ | 2,550 | -5.4 | 234.6k | Very Poor |
| **Overall** | **26,818** | **0.115** | **44.4k** | **Moderate** |

*Note: Negative R² indicates predictions worse than mean, showing poor local fit*

---

## 💡 Key Lessons

### ✅ What Works
- Full dataset preservation
- Hyperparameter optimization
- Feature engineering
- Ensemble methods
- Adding informative features (+1.7%)

### ❌ What Doesn't Work
- Filtering to subsets (overfitting)
- Over-weighting minorities (too small signal)
- Training on isolated segments (loss of information)

---

## 🎓 Conclusion

The professor's recommendation was valuable for understanding model behavior:
- ✅ Correctly identified that model struggles with certain patterns
- ❌ Data curation filtering is not the solution
- → Root cause: insufficient features, not bad data
- → Solution: better features, more data, or alternative algorithms

**Final recommendation:** Keep baseline GradientBoosting model with full data.
For significant improvements (R²>0.25), pursue:
1. Better feature engineering
2. More/different data (temporal, behavioral)
3. Advanced algorithms (XGBoost, LightGBM, Neural Networks)

---

## 📞 Questions?

Check:
1. ANALYSIS_REPORT.md for detailed explanation
2. SUMMARY_AND_NEXT_STEPS.txt for next steps
3. Individual analysis scripts for specific experiments
4. CSV results files for comparative metrics

