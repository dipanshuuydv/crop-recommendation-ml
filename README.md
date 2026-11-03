# Crop Recommendation using Machine Learning

A classification project that recommends the most suitable crop from soil nutrients and weather conditions. I built it to practise the full data-to-model workflow: cleaning and exploring data, comparing models fairly, and interpreting the results.

## Dataset

Kaggle "Crop Recommendation Dataset" (`Crop_recommendation.csv`).

- 2,200 rows, 8 columns
- 7 features: `N`, `P`, `K` (soil nutrients), `temperature`, `humidity`, `ph`, `rainfall`
- Target: `label` (22 crops, 100 rows each, so the classes are balanced)
- No missing values

## Approach

1. **Inspect and clean:** checked shape, data types, missing values and value ranges with Pandas.
2. **Explore:** feature histograms, correlation heatmap and average rainfall per crop (Matplotlib, Seaborn).
3. **Split:** 80/20 stratified train-test split (1,760 train rows, 440 test rows, `random_state=42`).
4. **Train three models:** Logistic Regression (with feature scaling), Decision Tree, Random Forest (200 trees).
5. **Evaluate:** accuracy, macro precision, macro recall, macro F1 and 5-fold cross-validation, plus a confusion matrix and feature importance.

## Results

| Model | Accuracy | Precision | Recall | F1 | 5-fold CV Accuracy |
|---|---|---|---|---|---|
| Random Forest | 0.9955 | 0.9957 | 0.9955 | 0.9955 | 0.9945 |
| Decision Tree | 0.9795 | 0.9806 | 0.9795 | 0.9794 | 0.9868 |
| Logistic Regression | 0.9727 | 0.9740 | 0.9727 | 0.9725 | 0.9714 |

**Best model: Random Forest, 99.55% test accuracy (99.45% cross-validated).**

The test accuracy and cross-validated accuracy are close, so the result does not depend on one lucky split.

## Key Insights

- **Rainfall is the most important feature** in both the Decision Tree and the Random Forest. Humidity and potassium (`K`) follow in the Random Forest. `ph` matters least.
- **Phosphorus (`P`) and potassium (`K`) are strongly correlated** (0.74), the strongest relationship among the features.
- **Crops need very different conditions.** For example, rice has the highest average rainfall of all 22 crops, which is why rainfall separates crops so well.
- **Only 2 of 440 test rows were misclassified** by the Random Forest: one rice sample predicted as jute, and one blackgram sample predicted as maize.

## Limitations

- This dataset is clean and the crops are well separated, which is why accuracy is very high. Real farm data would be noisier, so I would not expect the same score outside this dataset.
- The model predicts from the seven features only. It does not consider soil type, season, market price or location.
- Hyperparameters were mostly left at defaults. I did not tune them.

## Tech Stack

Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Jupyter

## How to Run

```
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Open `crop_analysis.ipynb` and run all cells. Keep `Crop_recommendation.csv` in the same folder.

## Files

- `crop_analysis.ipynb`: the full analysis and model comparison
- `Crop_recommendation.csv`: dataset
- `model_comparison.csv`: results table

## Author

Dipanshu Kumar Yadav, BCA student (2027), AISECT University, Hazaribagh.
