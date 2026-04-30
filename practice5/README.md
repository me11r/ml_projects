# Practice 5 - Ensemble Methods for Classification
## Task 4 Variant A: Extra Trees Classifier

### Overview
This notebook demonstrates ensemble learning methods applied to the Adult Census Income Dataset. The main focus is on **Extra Trees Classifier (Variant A)** compared with other ensemble methods and a baseline Decision Tree model.

### Dataset
- **Name**: Adult Census Income Dataset  
- **Samples**: 32,562 records
- **Target**: Income level (<=50K or >50K)
- **Features**: 14 features (age, workclass, education, occupation, etc.)

### Models Implemented
1. **Baseline Model**: Decision Tree Classifier
2. **Ensemble Models**:
   - Random Forest (100 estimators)
   - **Extra Trees Classifier (100 estimators)** ← VARIANT A
   - Gradient Boosting (100 estimators)

### Notebook Structure

#### Step 1: Data Loading & Exploration
- Load Adult Census Income Dataset
- Explore dataset structure and distributions
- Check target variable balance

#### Step 2: Data Preparation
- Handle missing values (replace '?' with mode)
- Encode categorical features using LabelEncoder
- Prepare features (X) and target (y)
- Split data: 80% training, 20% testing (stratified)

#### Step 3: Baseline Model - Decision Tree
- Train basic Decision Tree (max_depth=10)
- Generate predictions on train/test sets
- Calculate training and testing accuracy

#### Step 4: Ensemble Models
- **Random Forest**: 100 trees with voting
- **Extra Trees**: 100 trees with randomized feature selection (VARIANT A)
- **Gradient Boosting**: Sequential model building with learning rate=0.1

#### Step 5: Evaluation
- Confusion matrices for all models
- Classification reports (precision, recall, F1-score)
- Detailed analysis of True Positives/False Positives/etc.

#### Step 6: Model Comparison
- Accuracy comparison table
- Precision, Recall, and F1-score metrics
- Feature importance analysis from ensemble models

#### Step 7: Visualization (4 Charts)
1. Training vs Testing Accuracy comparison
2. Confusion Matrix heatmaps for all models
3. Precision/Recall/F1 comparison
4. Feature Importance comparison (Top 10 features)

#### Step 8: Conclusion
- Summary of findings
- Best performing model identification
- Key insights about ensemble methods

### Key Features of the Notebook
✓ **45 total cells** (37 code cells + 8 markdown sections)  
✓ **Comprehensive data preparation** with proper handling of missing values  
✓ **Multiple ensemble methods** for comparison  
✓ **Extra Trees Classifier** (Variant A) as the main focus  
✓ **Extensive evaluation metrics** (Accuracy, Precision, Recall, F1-Score)  
✓ **4 visualizations** for model comparison  
✓ **Feature importance analysis** from multiple models  
✓ **Professional conclusions** and learning outcomes  

### Why Extra Trees Classifier?
Extra Trees (Extremely Randomized Trees) differs from Random Forest in:
- Uses **randomized thresholds** for feature selection instead of optimal splits
- Typically **faster training** due to randomization
- **Reduced variance** through more aggressive randomization
- Can improve generalization in certain datasets

### How to Run
```python
# The notebook is self-contained and ready to run
# Open main.ipynb in Jupyter Notebook or JupyterLab
# Execute cells sequentially from top to bottom
```

### Expected Results
- Extra Trees shows comparable or better performance than Random Forest
- Ensemble methods outperform the baseline Decision Tree
- Reduced overfitting gap between train/test sets
- Detailed feature importance rankings from each model

### Files
- `main.ipynb` - Main Jupyter notebook with all steps
- `adult.csv` - Input dataset
- `README.md` - This file

---
**Course**: Machine Learning Practice 5  
**Task**: Task 4 - Variant A: Extra Trees Classifier  
**Completion Date**: 2026-04-28
