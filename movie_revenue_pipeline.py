import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import ast
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Setting aesthetic parameters for plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_data(file_path):
    print("--- Step 2: Loading Dataset ---")
    df = pd.read_csv(file_path)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nDataset Info:")
    df.info()
    print("\nStatistical Summary:")
    print(df.describe())
    return df

def clean_data(df):
    print("\n--- Step 3: Data Cleaning ---")
    
    # Drop unnecessary columns
    cols_to_drop = ['id', 'title', 'production_companies', 'production_countries', 'cast', 'director']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Handle missing values
    # For numeric columns, fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Replace budget = 0 with median of non-zero budgets
    median_budget = df[df['budget'] > 0]['budget'].median()
    df['budget'] = df['budget'].replace(0, median_budget)
    
    # Convert release_date and extract features
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df = df.dropna(subset=['release_date']) # Drop rows where date is invalid
    df['release_year'] = df['release_date'].dt.year
    df['release_month'] = df['release_date'].dt.month
    df = df.drop(columns=['release_date'])
    
    return df

def feature_engineering(df):
    print("\n--- Step 4: Feature Engineering ---")
    
    # Parse genres (stored as stringified lists)
    def parse_genres(genre_str):
        try:
            genres = ast.literal_eval(genre_str)
            return genres if isinstance(genres, list) else []
        except:
            return []

    if 'genres' in df.columns:
        df['genres_list'] = df['genres'].apply(parse_genres)
        
        # One-hot encoding for top genres
        all_genres = [genre for sublist in df['genres_list'] for genre in sublist]
        top_genres = pd.Series(all_genres).value_counts().head(10).index.tolist()
        
        for genre in top_genres:
            df[f'genre_{genre}'] = df['genres_list'].apply(lambda x: 1 if genre in x else 0)
        
        df = df.drop(columns=['genres', 'genres_list'])

    # One-hot encoding for original_language (top 5 only to avoid sparse matrix)
    if 'original_language' in df.columns:
        top_langs = df['original_language'].value_counts().head(5).index.tolist()
        df['original_language'] = df['original_language'].apply(lambda x: x if x in top_langs else 'other')
        df = pd.get_dummies(df, columns=['original_language'], prefix='lang')

    # Log transformation for skewed financial features
    df['log_budget'] = np.log1p(df['budget'])
    df['log_revenue'] = np.log1p(df['revenue'])
    
    return df

def perform_eda(df):
    print("\n--- Step 5: Exploratory Data Analysis ---")
    
    # Correlation Heatmap
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.savefig('correlation_heatmap.png')
    plt.close()

    # Budget vs Revenue
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='budget', y='revenue', alpha=0.5)
    plt.title('Budget vs Revenue')
    plt.savefig('budget_vs_revenue.png')
    plt.close()

    # Popularity vs Revenue
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='popularity', y='revenue', alpha=0.5, color='orange')
    plt.title('Popularity vs Revenue')
    plt.savefig('popularity_vs_revenue.png')
    plt.close()

    # Revenue Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['revenue'], bins=50, kde=True)
    plt.title('Revenue Distribution')
    plt.savefig('revenue_distribution.png')
    plt.close()
    
    print("EDA plots saved as PNG files.")

def train_and_evaluate(df):
    print("\n--- Step 6: Data Preprocessing ---")
    
    # Drop raw budget and revenue, keep log versions for training
    # We will use log_revenue as the target for better model performance
    X = df.drop(columns=['revenue', 'log_revenue', 'budget'])
    y = df['log_revenue']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Train set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")

    print("\n--- Step 7 & 8: Model Training & Evaluation (Baseline) ---")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    
    # Evaluate logged targets
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print("\nBaseline Model Performance (on Log Revenue):")
    print(f"R2 Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    return X_train, X_test, y_train, y_test, rf

def tune_hyperparameters(X_train, y_train):
    print("\n--- Step 9: Hyperparameter Tuning ---")
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def final_results(best_model, X_test, y_test, feature_names):
    print("\n--- Step 10 & 11: Final Performance & Feature Importance ---")
    y_pred = best_model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    results = pd.DataFrame({
        'Metric': ['R2 Score', 'RMSE', 'MAE'],
        'Value': [r2, rmse, mae]
    })
    print("\nTuned Model Performance:")
    print(results)

    # Feature Importance
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=importances[indices][:15], y=np.array(feature_names)[indices][:15])
    plt.title('Top 15 Feature Importances')
    plt.savefig('feature_importance.png')
    plt.close()
    
    # Prediction Visualization
    plt.figure(figsize=(10, 10))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
    plt.xlabel('Actual (Log Revenue)')
    plt.ylabel('Predicted (Log Revenue)')
    plt.title('Actual vs Predicted Revenue (Log Scale)')
    plt.savefig('actual_vs_predicted.png')
    plt.close()
    
    print("Final visualization plots saved as PNG files.")
    
    return best_model

def save_model(model):
    print("\n--- Step 14: Saving the Model ---")
    joblib.dump(model, 'movie_revenue_model.pkl')
    print("Model saved to 'movie_revenue_model.pkl'")

if __name__ == "__main__":
    DATA_PATH = 'moviedata.csv'
    
    # Pipeline Execution
    data = load_data(DATA_PATH)
    data = clean_data(data)
    data = feature_engineering(data)
    perform_eda(data)
    X_train, X_test, y_train, y_test, rf_baseline = train_and_evaluate(data)
    
    # Best model selection (using tuning)
    best_rf = tune_hyperparameters(X_train, y_train)
    final_model = final_results(best_rf, X_test, y_test, X_train.columns.tolist())
    
    save_model(final_model)
    print("\nPipeline execution complete.")
