# Data Science Koans - Complete Catalog

**Total Koans: 130 across 15 notebooks**

This comprehensive document specifies every koan in the Data Science Koans project, organized by notebook and difficulty level.

---

## 📘 Notebook 01: NumPy Fundamentals (10 koans)

**Prerequisites**: Basic Python  
**Difficulty**: Beginner  
**Time**: 2-3 hours

| # | Title | Concept | Key Skills | Exercise Summary |
|---|-------|---------|------------|------------------|
| 1.1 | Array Creation | Creating 1D arrays | `np.array()` | Create array [1,2,3,4,5] |
| 1.2 | Multi-dimensional Arrays | 2D arrays/matrices | Nested lists to arrays | Create 3x3 matrix |
| 1.3 | Array Properties | Shape, dtype, ndim, size | Inspecting arrays | Determine array attributes |
| 1.4 | Array Creation Functions | Generation functions | `zeros()`, `ones()`, `arange()`, `linspace()` | Create arrays without explicit values |
| 1.5 | Array Indexing | Element access | Positive/negative indexing | Extract specific elements |
| 1.6 | Array Slicing | Subarrays | `start:stop:step` | Slice array ranges |
| 1.7 | 2D Indexing | Matrix access | `arr[row, col]` | Extract rows, columns, blocks |
| 1.8 | Array Operations | Element-wise arithmetic | `+`, `-`, `*`, `/`, `**` | Vectorized calculations |
| 1.9 | Broadcasting | Different shape operations | Broadcasting rules | Add 1D to 2D array |
| 1.10 | Array Methods | Aggregations | `mean()`, `sum()`, `std()`, axis param | Statistics along axes |

---

## 📘 Notebook 02: Pandas Essentials (10 koans)

**Prerequisites**: NumPy Fundamentals  
**Difficulty**: Beginner  
**Time**: 2-3 hours

| # | Title | Concept | Key Skills | Exercise Summary |
|---|-------|---------|------------|------------------|
| 2.1 | Creating Series | 1D labeled data | `pd.Series()`, index/values | Create Series with custom index |
| 2.2 | Series Operations | Vectorized ops | Arithmetic, boolean indexing | Filter and transform Series |
| 2.3 | Creating DataFrames | 2D labeled data | `pd.DataFrame()` from various sources | Create from dict, lists, arrays |
| 2.4 | DataFrame Properties | Structure inspection | `shape`, `columns`, `index`, `dtypes` | Inspect attributes |
| 2.5 | Column Selection | Accessing columns | `df['col']`, `df[['col1', 'col2']]` | Select single/multiple columns |
| 2.6 | Row Selection | Label/position indexing | `loc[]`, `iloc[]` | Select rows both ways |
| 2.7 | Boolean Indexing | Conditional filtering | Boolean masks, conditions | Filter with complex conditions |
| 2.8 | Basic Statistics | Descriptive stats | `describe()`, `mean()`, `value_counts()` | Calculate summaries |
| 2.9 | GroupBy Operations | Split-apply-combine | `groupby()`, aggregations | Group and aggregate data |
| 2.10 | Sorting and Ranking | Ordering data | `sort_values()`, `sort_index()`, `rank()` | Sort by multiple columns |

---

## 📘 Notebook 03: Data Exploration (10 koans)

**Prerequisites**: Pandas Essentials  
**Difficulty**: Beginner  
**Time**: 2-3 hours

| # | Title | Concept | Key Skills | Exercise Summary |
|---|-------|---------|------------|------------------|
| 3.1 | Loading CSV Data | File reading | `pd.read_csv()`, delimiters | Load and inspect CSV |
| 3.2 | Data Profiling | Dataset characteristics | `head()`, `info()`, `describe()` | Generate data profile |
| 3.3 | Missing Value Detection | Null identification | `isnull()`, `isna()`, `notna()` | Find and report missingness |
| 3.4 | Data Type Conversion | Dtype management | `astype()`, `to_numeric()`, `to_datetime()` | Convert to appropriate types |
| 3.5 | Basic Visualization | Simple plots | `plot()`, `hist()`, `scatter()` | Create exploratory viz |
| 3.6 | Correlation Analysis | Variable relationships | `corr()`, correlation matrix | Calculate correlations |
| 3.7 | Categorical Exploration | Category analysis | `value_counts()`, `unique()`, `nunique()` | Explore categorical distributions |
| 3.8 | Cross-tabulation | Frequency tables | `pd.crosstab()`, `pivot_table()` | Create and interpret cross-tabs |
| 3.9 | Outlier Detection | Extreme value identification | IQR method, z-scores, box plots | Detect outliers statistically |
| 3.10 | Data Quality Report | Comprehensive check | Combining multiple checks | Generate quality report |

---

## 📘 Notebook 04: Data Cleaning (10 koans)

**Prerequisites**: Data Exploration  
**Difficulty**: Beginner-Intermediate  
**Time**: 2-3 hours

| # | Title | Concept | Key Skills | Exercise Summary |
|---|-------|---------|------------|------------------|
| 4.1 | Missing Values - Deletion | Removing nulls | `dropna()`, `how`, `thresh`, `subset` | Strategic row/column removal |
| 4.2 | Missing Values - Imputation | Filling nulls | `fillna()`, forward/backward fill | Fill with mean, median, mode |
| 4.3 | Advanced Imputation | Sophisticated filling | Interpolation, KNN imputation | Use sklearn for imputation |
| 4.4 | Duplicate Detection | Finding duplicates | `duplicated()`, `drop_duplicates()` | Identify and remove duplicates |
| 4.5 | String Cleaning | Text normalization | `str` methods, regex | Clean and standardize strings |
| 4.6 | Data Type Validation | Type consistency | Schema validation, assertions | Ensure correct data types |
| 4.7 | Outlier Handling | Managing extremes | Capping, transformation, removal | Handle outliers appropriately |
| 4.8 | Inconsistent Categories | Category fixing | Mapping, replacement | Standardize categorical values |
| 4.9 | Data Validation Rules | Business rules | Custom validators | Implement domain constraints |
| 4.10 | Cleaning Pipeline | Complete workflow | Chaining operations | Build comprehensive cleaning pipeline |

---

## 📘 Notebook 05: Data Transformation (10 koans)

**Prerequisites**: Data Cleaning  
**Difficulty**: Beginner-Intermediate  
**Time**: 2-3 hours

| # | Title | Concept | Key Skills | Exercise Summary |
|---|-------|---------|------------|------------------|
| 5.1 | Scaling - StandardScaler | Z-score normalization | `StandardScaler`, fit/transform | Scale to mean=0, std=1 |
| 5.2 | Scaling - MinMaxScaler | Range normalization | `MinMaxScaler` | Scale to [0,1] range |
| 5.3 | Robust Scaling | Outlier-resistant scaling | `RobustScaler` | Scale using median and IQR |
| 5.4 | Label Encoding | Ordinal encoding | `LabelEncoder` | Convert categories to integers |
| 5.5 | One-Hot Encoding | Categorical to binary | `pd.get_dummies()`, `OneHotEncoder` | Create binary columns |
| 5.6 | Ordinal Encoding | Ordered categories | `OrdinalEncoder` | Encode with order preservation |
| 5.7 | Log Transformation | Skewness reduction | `np.log()`, `np.
log1p()` | Apply log transforms |
| 5.8 | Power Transformation | Non-linear transforms | `PowerTransformer`, Box-Cox | Transform to normality |
| 5.9 | Binning/Discretization | Continuous to categorical | `pd.cut()`, `pd.qcut()` | Create bins from continuous |
| 5.10 | Polynomial Features | Feature interactions | `PolynomialFeatures` | Generate polynomial terms |

---

## 📘 Notebook 06: Feature Engineering Basics (10 koans)

**Prerequisites**: Data Transformation  
**Difficulty**: Intermediate  
**Time**: 3-4 hours

| # | Title | Concept | Key Skills | Exercise Summary |
|---|-------|---------|------------|------------------|
| 6.1 | Date/Time Features | Temporal extraction | `dt` accessor, components | Extract year, month, day, hour |
| 6.2 | Cyclical Features | Circular encoding | Sin/cos transforms | Encode hour, month cyclically |
| 6.3 | Text Features - Basics | String features | Length, word count, patterns | Extract text statistics |
| 6.4 | Text Features - TF-IDF | Text vectorization | `TfidfVectorizer` | Convert text to numeric |
| 6.5 | Aggregation Features | Group statistics | `groupby()` + `transform()` | Create group-level features |
| 6.6 | Lag Features | Time series lags | `shift()` | Create previous values |
| 6.7 | Rolling Statistics | Moving windows | `rolling()` | Calculate moving averages |
| 6.8 | Interaction Features | Feature combinations | Multiplication, ratios | Create feature interactions |
| 6.9 | Feature Selection - Filter | Statistical selection | Correlation, variance threshold | Select by statistics |
| 6.10 | Feature Selection - Wrapper | Model-based selection | `SelectKBest`, `RFE` | Select using models |

---

## 📘 Notebook 07: Regression Basics (10 koans)

**Prerequisites**: Feature Engineering Basics  
**Difficulty**: Intermediate  
**Time**: 3-4 hours

| # | Title | Concept | Key Skills | Exercise Summary |
|---|-------|---------|------------|------------------|
| 7.1 | Linear Regression | Basic regression | `LinearRegression`, fit/predict | Fit simple linear model |
| 7.2 | Multiple Regression | Multiple predictors | Multiple features | Use multiple X variables |
| 7.3 | Regression Metrics | Performance evaluation | MSE, RMSE, MAE, R² | Calculate regression metrics |
| 7.4 | Train-Test Split | Data splitting | `train_test_split()` | Split data properly |
| 7.5 | Ridge Regression | L2 regularization | `Ridge`, alpha parameter | Apply L2 penalty |
| 7.6 | Lasso Regression | L1 regularization | `Lasso`, feature selection | Apply L1 penalty |
| 7.7 | ElasticNet | Combined regularization | `ElasticNet`, l1_ratio | Combine L1 and L2 |
| 7.8 | Polynomial Regression | Non-linear relationships | `PolynomialFeatures` + regression | Fit polynomial models |
| 7.9 | Residual Analysis | Error examination | Residual plots, normality | Analyze model residuals |
| 7.10 | Prediction Intervals | Uncertainty quantification | Confidence intervals | Estimate prediction uncertainty |

---

## 📘 Notebook 08: Classification Basics (10 koans)

**Prerequisites**: Regression Basics  
**Difficulty**: Intermediate  
**Time**: 3-4 hours

| # | Title | Concept | Key Skills | Exercise Summary |
|---|-------|---------|------------|------------------|
| 8.1 | Logistic Regression | Binary classification | `LogisticRegression` | Classify binary outcomes |
| 8.2 | Classification Metrics | Performance measures | Accuracy, precision, recall, F1 | Calculate classification metrics |
| 8.3 | Confusion Matrix | Error analysis | `confusion_matrix`, interpretation | Analyze prediction errors |
| 8.4 | ROC Curve and AUC | Threshold analysis | `roc_curve`, `roc_auc_score` | Plot and interpret ROC |
| 8.5 | Decision Trees | Tree-based classification | `DecisionTreeClassifier`, depth | Build decision tree |
| 8.6 | K-Nearest Neighbors | Instance-based learning | `KNeighborsClassifier`, k parameter | Classify using neighbors |
| 8.7 | Naive Bayes | Probabilistic classification | `GaussianNB`, `MultinomialNB` | Apply Bayes theorem |
| 8.8 | Support Vector Machines | Margin maximization | `SVC`, kernel parameter | Use SVM for classification |
| 8.9 | Multi-class Classification | Multiple classes | One-vs-rest, one-vs-one | Handle 3+ classes |
| 8.10 | Class Imbalance | Handling imbalanced data | `class_weight`, SMOTE | Deal with imbalanced classes |

---

## 📘 Notebook 09: Model Evaluation (10 koans)

**Prerequisites**: Classification Basics  
**Difficulty**: Intermediate  
**Time**: 3-4 hours

| # | Title | Concept | Key Skills | Exercise Summary |
|---|-------|---------|------------|------------------|
| 9.1 | Cross-Validation | Robust evaluation | `cross_val_score`, k-fold | Perform k-fold CV |
| 9.2 | Stratified CV | Preserving distributions | `StratifiedKFold` | Maintain class proportions |
| 9.3 | Learning Curves | Training dynamics | `learning_curve` | Plot learning curves |
| 9.4 | Validation Curves | Hyperparameter effects | `validation_curve` | Analyze parameter impact |
| 9.5 | Bias-Variance Tradeoff | Model complexity | Underfitting vs overfitting | Understand tradeoff |
| 9.6 | Overfitting Detection | Generalization check | Train vs test performance | Identify overfitting |
| 9.7 | Model Comparison | Multiple models | Statistical tests | Compare model performance |
| 9.8 | Nested Cross-Validation | Unbiased evaluation | Nested CV loops | Proper model selection |
| 9.9 | Custom Scoring | Domain metrics | `make_scorer` | Create custom metrics |
| 9.10 | Model Persistence | Saving models | `joblib`, `pickle` | Save and load models |

---

## 📘 Notebook 10: Clustering (8 koans)

**Prerequisites**: Model Evaluation  
**Difficulty**: Intermediate-Advanced  
**Time**: 2-3 hours

| # | Title | Concept | Key Skills | Exercise Summary |
|---|-------|---------|------------|------------------|
| 10.1 | K-Means Clustering | Centroid-based clustering | `KMeans`, n_clusters | Perform k-means |
| 10.2 | Elbow Method | Optimal k selection | Inertia, elbow plot | Find optimal clusters |
| 10.3 | Silhouette Analysis | Cluster quality | `silhouette_score`, `silhouette_samples` | Evaluate clustering |
| 10.4 | Hierarchical Clustering | Dendrogram-based | `AgglomerativeClustering`, linkage | Build hierarchy |
| 10.5 | DBSCAN | Density-based clustering | `DBSCAN`, eps, min_samples | Find arbitrary shapes |
| 10.6 | Gaussian Mixture Models | Probabilistic clustering | `GaussianMixture` | Soft clustering |
| 10.7 | Cluster Visualization | Dimensionality reduction | PCA for visualization | Visualize clusters |
| 10.8 | Cluster Interpretation | Understanding clusters | Profiling, characterization | Interpret cluster meaning |

---

## 📘 Notebook 11: Dimensionality Reduction (8 koans)

**Prerequisites**: Clustering  
**Difficulty**: Intermediate-Advanced  
**Time**: 2-3 hours

| # | Title | Concept | Key Skills | Exercise Summary |
|---|-------|---------|------------|------------------|
| 11.1 | Principal Component Analysis | Linear reduction | `PCA`, components | Reduce dimensions with PCA |
| 11.2 | Explained Variance | Information retention | `explained_variance_ratio_` | Analyze variance explained |
| 11.3 | Scree Plot | Component selection | Plotting variance | Choose number of components |
| 11.4 | PCA for Visualization | 2D/3D projection | Visualization in low dimensions | Project to 2D/3D |
| 11.5 | Feature Loadings | Component interpretation | Loading vectors | Understand PC meaning |
| 11.6 | t-SNE | Non-linear reduction | `TSNE`, perplexity | Visualize with t-SNE |
| 11.7 | UMAP | Modern reduction | `umap.UMAP` | Use UMAP for visualization |
| 11.8 | Dimensionality Curse | High-dimensional problems | Distance metrics, sparsity | Understand curse of dimensionality |

---

## 📘 Notebook 12: Ensemble Methods (7 koans)

**Prerequisites**: Dimensionality Reduction  
**Difficulty**: Advanced  
**Time**: 2-3 hours

| # | Title | Concept | Key Skills | Exercise Summary |
|---|-------|---------|------------|------------------|
| 12.1 | Random Forest | Bagging trees | `RandomForestClassifier/Regressor` | Build random forest |
| 12.2 | Feature Importance | Variable significance | `feature_importances_` | Rank feature importance |
| 12.3 | Gradient Boosting | Sequential boosting | `GradientBoostingClassifier/Regressor` | Apply gradient boosting |
| 12.4 | XGBoost | Advanced boosting | `xgboost.XGBClassifier/Regressor` | Use XGBoost |
| 12.5 | Voting Classifier | Model combination | `VotingClassifier`, hard/soft voting | Combine multiple models |
| 12.6 | Stacking | Meta-learning | `StackingClassifier/Regressor` | Stack models |
| 12.7 | Ensemble Comparison | Method evaluation | Comparing ensemble techniques | Compare ensemble methods |

---

## 📘 Notebook 13: Hyperparameter Tuning (7 koans)

**Prerequisites**: Ensemble Methods  
**Difficulty**: Advanced  
**Time**: 2-3 hours

| # | Title | Concept | Key Skills | Exercise Summary |
|---|-------|---------|------------|------------------|
| 13.1 | Grid Search | Exhaustive search | `GridSearchCV` | Search parameter grid |
| 13.2 | Random Search | Random sampling | `RandomizedSearchCV` | Random parameter search |
| 13.3 | Parameter Distributions | Search spaces | Defining distributions | Specify parameter ranges |
| 13.4 | Bayesian Optimization | Smart search | `BayesSearchCV` (scikit-optimize) | Use Bayesian optimization |
| 13.5 | Early Stopping | Training efficiency | Monitoring validation | Stop training early |
| 13.6 | Nested CV for Tuning | Unbiased tuning | Nested cross-validation | Tune without bias |
| 13.7 | AutoML Basics | Automated ML | `auto-sklearn` or `TPOT` | Automated model selection |

---

## 📘 Notebook 14: Model Selection and Pipelines (5 koans)

**Prerequisites**: Hyperparameter Tuning  
**Difficulty**: Advanced  
**Time**: 2 hours

| # | Title | Concept | Key Skills | Exercise Summary |
|---|-------|---------|------------|------------------|
| 14.1 | Pipeline Basics | Workflow automation | `Pipeline`, chaining steps | Build basic pipeline |
| 14.2 | ColumnTransformer | Feature-specific transforms | `ColumnTransformer` | Transform different columns |
| 14.3 | Custom Transformers | Extending sklearn | `BaseEstimator`, `TransformerMixin` | Create custom transformer |
| 14.4 | Pipeline with GridSearch | Integrated tuning | Pipeline + GridSearchCV | Tune entire pipeline |
| 14.5 | Production Pipeline | Deployment-ready | Complete workflow | Build production pipeline |

---

## 📘 Notebook 15: Ethics and Bias (5 koans)

**Prerequisites**: Model Selection and Pipelines  
**Difficulty**: Advanced  
**Time**: 2 hours

| # | Title | Concept | Key Skills | Exercise Summary |
|---|-------|---------|------------|------------------|
| 15.1 | Fairness Metrics | Bias measurement | Demographic parity, equal opportunity | Calculate fairness metrics |
| 15.2 | Bias Detection | Identifying bias | Group-based analysis | Detect model bias |
| 15.3 | Bias Mitigation | Reducing bias | Reweighting, threshold adjustment | Mitigate identified bias |
| 15.4 | Model Interpretability | Explainability | SHAP, LIME | Explain model predictions |
| 15.5 | Responsible ML Checklist | Best practices | Documentation, monitoring | Create ML ethics checklist |

---

## Summary Statistics

### By Difficulty Level
- **Beginner**: 30 koans (Notebooks 1-3)
- **Beginner-Intermediate**: 20 koans (Notebooks 4-5)
- **Intermediate**: 40 koans (Notebooks 6-9)
- **Intermediate-Advanced**: 16 koans (Notebooks 10-11)
- **Advanced**: 24 koans (Notebooks 12-15)

### By Topic Area
- **NumPy/Pandas Fundamentals**: 20 koans
- **Data Preparation**: 30 koans
- **Supervised Learning**: 30 koans
- **Unsupervised Learning**: 16 koans
- **Advanced Techniques**: 24 koans
- **Best Practices**: 10 koans

### Estimated Total Time
- **Beginner Path** (Notebooks 1-3): 6-9 hours
- **Intermediate Path** (Notebooks 4-9): 15-21 hours
- **Advanced Path** (Notebooks 10-15): 12-16 hours
- **Complete Course**: 33-46 hours

---

## Implementation Priority

### Phase 1 (MVP - Weeks 1-4)
1. ✅ Notebook 01: NumPy Fundamentals
2. Notebook 02: Pandas Essentials
3. Notebook 03: Data Exploration

### Phase 2 (Core Content - Weeks 5-8)
4. Notebook 04: Data Cleaning
5. Notebook 05: Data Transformation
6. Notebook 07: Regression Basics
7. Notebook 08: Classification Basics

### Phase 3 (Advanced Content - Weeks 9-12)
8. Notebook 06: Feature Engineering Basics
9. Notebook 09: Model Evaluation
10. Notebook 10: Clustering
11. Notebook 12: Ensemble Methods

### Phase 4 (Expert Content - Weeks 13-14)
12. Notebook 11: Dimensionality Reduction
13. Notebook 13: Hyperparameter Tuning
14. Notebook 14: Model Selection and Pipelines
15. Notebook 15: Ethics and Bias

---

## Next Steps

1. **Complete Notebook 01** with all 10 koans (currently has 5 demo koans)
2. **Create Notebook 02** (Pandas Essentials)
3. **Create Notebook 03** (Data Exploration)
4. **Test learning flow** with users
5. **Iterate based on feedback**
6. **Continue with remaining notebooks**

---

## Notes

- Each koan should take 10-20 minutes to complete
- Koans build progressively within each notebook
- Later notebooks assume mastery of earlier concepts
- Real datasets introduced starting from Notebook 7
- Ethics and bias integrated throughout, with