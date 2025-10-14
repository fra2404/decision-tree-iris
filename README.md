# Ensemble Learning on Iris Dataset

**Course:** Artificial Intelligence  
**Assignment:** 3 - Ensemble Learning  
**Date:** October 2025  
**Student:** Francesco

## Project Overview

This project implements various ensemble learning methods to solve classification problems on the classic Iris dataset. The implementation explores different ensemble techniques including voting, bagging, stacking, and random forests.

## Dataset

**Iris Dataset** (from sklearn.datasets)

- **Samples:** 150 instances
- **Features:** 4 numerical features (sepal length, sepal width, petal length, petal width)
- **Classes:** 3 species (setosa, versicolor, virginica)
- **Task:** Multi-class classification

## Ensemble Methods Implemented

1. **Voting Classifier**

   - Hard voting and soft voting
   - Combination of different base classifiers

2. **Bagging**

   - Bootstrap aggregating with decision trees
   - Random subsampling techniques

3. **Random Forest**

   - Advanced bagging with feature randomness
   - Out-of-bag error estimation

4. **Stacking**
   - Meta-learner approach
   - Multiple levels of learning

## Project Structure

```
Assignment/3/
├── README.md                   # Project documentation
├── src/                        # Source code
│   ├── ensemble_core.py        # Core ensemble implementations
│   ├── data_processor.py       # Data loading and preprocessing
│   ├── visualization.py        # Plotting and analysis tools
│   └── main_iris_ensemble.py   # Main application
├── results/                    # Generated results
│   ├── *.csv                  # Performance metrics
│   ├── *.png                  # Visualizations
│   └── *.txt                  # Analysis reports
└── report/                     # LaTeX documentation
    ├── iris_ensemble_report.tex
    └── iris_ensemble_report.pdf
```

## Key Features

- ✅ **Modular Architecture:** Clean separation of concerns
- ✅ **Multiple Ensemble Methods:** Comprehensive implementation
- ✅ **Advanced Visualization:** Decision boundaries and performance plots
- ✅ **Statistical Analysis:** Cross-validation and confidence intervals
- ✅ **Professional Documentation:** LaTeX report with academic standards

## Requirements

- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## Usage

```bash
# Run the main ensemble analysis
python src/main_iris_ensemble.py

# Results will be generated in the results/ directory
# Report compilation available in report/ directory
```

## Performance Targets

- **Accuracy:** >95% on test set
- **Robustness:** Consistent performance across methods
- **Interpretability:** Clear feature importance analysis
- **Efficiency:** Optimized ensemble sizes

## Academic Context

This project demonstrates understanding of:

- Ensemble learning principles and theory
- Bias-variance trade-off in machine learning
- Cross-validation and model evaluation
- Feature engineering and selection
- Statistical significance testing

---

_Built upon the modular architecture from Assignment 2 (ID3 Decision Trees)_
