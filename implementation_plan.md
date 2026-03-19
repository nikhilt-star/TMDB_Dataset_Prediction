# Random Forest Model on Movie Data

Apply Random Forest to predict movie success (vote_average >= 7.0) based on features like budget, revenue, runtime, popularity, and genres.

## Proposed Changes

### [NEW] [train_model.py](file:///c:/Users/Admin/OneDrive/Desktop/ML%20LEARN/train_model.py)
A Python script to:
1. Load `moviedata.csv`.
2. Clean and preprocess data:
   - Handle missing values (fill or drop).
   - Create binary target `is_good` (1 if `vote_average >= 7.0`, else 0).
   - Encode categorical features (`genres`, `original_language`).
   - Select relevant numerical features (`budget`, `revenue`, `runtime`, `popularity`, `vote_count`).
3. Split data into training and test sets.
4. Train a Random Forest Classifier.
5. Evaluate model with accuracy, confusion matrix, and classification report.
6. Display feature importance.

## Verification Plan

### Automated Tests
- Run `python train_model.py` and verify it prints accuracy and model performance metrics.
- Check that the model achieves a reasonable accuracy (e.g., > 70%).

### Manual Verification
- Review the generated classification report and confusion matrix.
- Check the feature importance plot/list to ensure it makes sense.
