import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# STEP 1: Load and Initial Cleaning
drop_columns = [
    'parcelid', 'assessmentyear', 'taxamount', 'taxdelinquencyflag', 'taxdelinquencyyear', 'censustractandblock',
    'rawcensustractandblock', 'latitude', 'longitude', 'finishedfloor1squarefeet', 'finishedsquarefeet13',
    'finishedsquarefeet15', 'finishedsquarefeet50', 'finishedsquarefeet6', 'buildingclasstypeid', 'storytypeid',
    'typeconstructiontypeid', 'hashottuborspa', 'fireplaceflag', 'pooltypeid10', 'pooltypeid2', 'pooltypeid7',
    'propertyzoningdesc', 'propertycountylandusecode', 'unitcnt', 'yardbuildingsqft17', 'yardbuildingsqft26'
]
df = pd.read_csv("T:\CS Journey\Projectss\AIML Work\Project\properties_2016.csv").drop(columns=drop_columns)
df_before = df.copy()

# STEP 2: Numeric Outlier Handling & Feature Engineering
numeric_cap_dict = {'roomcnt': 10, 'bedroomcnt': 6, 'bathroomcnt': 6,
                    'threequarterbathnbr': 3, 'fullbathcnt': 5, 'fireplacecnt': 3,
                    'poolcnt': 3, 'garagecarcnt': 4, 'numberofstories': 4, 'calculatedbathnbr': 5}
sqft_cap_cols = ['lotsizesquarefeet', 'calculatedfinishedsquarefeet', 'finishedsquarefeet12', 'garagetotalsqft', 'basementsqft', 'poolsizesum']
target_cap_cols = ['structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'landtaxvaluedollarcnt']

# Cap/clean numerics
for col, cap in numeric_cap_dict.items():
    if col in df.columns:
        df[col + '_clean'] = np.where(df[col] > cap, cap, df[col])
        df[col + '_clean'] = df[col + '_clean'].fillna(df[col].median())
for col in sqft_cap_cols + target_cap_cols:
    if col in df.columns:
        val_cap = df[col].quantile(0.99)
        df[col + '_clean'] = np.where(df[col] > val_cap, val_cap, df[col])
        df[col + '_clean'] = df[col + '_clean'].fillna(df[col].median())
        df[col + '_log1p'] = np.log1p(df[col + '_clean'])

# Feature engineering: Example - binary basement flag
if 'basementsqft_clean' in df.columns:
    df['has_basement'] = (df['basementsqft_clean'] > 0).astype(int)

# STEP 3: Categorical Encoding
one_hot_cols = ['airconditioningtypeid', 'heatingorsystemtypeid', 'decktypeid']
label_encode_cols = ['fips', 'propertylandusetypeid', 'regionidcity', 'regionidcounty', 'regionidneighborhood', 'regionidzip',
                     'architecturalstyletypeid', 'buildingqualitytypeid']

for col in one_hot_cols:
    if col in df.columns:
        df[col] = df[col].fillna(-1).astype(int).astype(str)
        df = pd.get_dummies(df, columns=[col], prefix=col)

for col in label_encode_cols:
    if col in df.columns:
        df[col] = df[col].fillna(-1).astype(int).astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# Year bound and imputation
if 'yearbuilt' in df.columns:
    df['yearbuilt_clean'] = np.where(df['yearbuilt'] < 1850, 1850, df['yearbuilt'])
    df['yearbuilt_clean'] = np.where(df['yearbuilt_clean'] > 2025, 2025, df['yearbuilt_clean'])
    df['yearbuilt_clean'] = df['yearbuilt_clean'].fillna(df['yearbuilt'].mode()[0])

# Final feature set: remove engineered price/value features
exclude_targets = [
    "taxvaluedollarcnt_clean", "taxvaluedollarcnt_log1p",
    "structuretaxvaluedollarcnt_clean", "structuretaxvaluedollarcnt_log1p",
    "landtaxvaluedollarcnt_clean", "landtaxvaluedollarcnt_log1p"
]
final_cols = [col for col in df if (col.endswith('_clean') or col.endswith('_log1p') or col in label_encode_cols 
                                    or any(col.startswith(prefix) for prefix in one_hot_cols) or col == 'has_basement') 
                                    and col not in exclude_targets]
df = df[final_cols + ['taxvaluedollarcnt_clean']]

# STEP 4: Sampling (for speed during dev)
df_sample = df.sample(n=200000, random_state=42)
X = df_sample[[col for col in df_sample.columns if col not in exclude_targets + ['taxvaluedollarcnt_clean']]]
y = df_sample['taxvaluedollarcnt_clean']

# STEP 5: Train-Test Split & Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
numeric_cols = X_train.select_dtypes(include='number').columns
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
print("Train shape:", X_train_scaled.shape)
print("Test shape:", X_test_scaled.shape)

# STEP 6: Model Training (Tuned XGBoost)
best_params = {
    'colsample_bytree': 0.7182534743350856,
    'gamma': 0.052747129915135305,
    'learning_rate': 0.10130691409658205,
    'max_depth': 6,
    'min_child_weight': 4,
    'n_estimators': 133,
    'subsample': 0.9570235993959911
}
xgb_best = XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1,
    **best_params
)
xgb_best.fit(X_train_scaled, y_train)

# STEP 7: Model Evaluation
preds = xgb_best.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)
print(f"Tuned XGBoost: R^2={r2:.4f}, RMSE={rmse:.0f}")

# STEP 8: Feature Importance Visualization
importances = xgb_best.feature_importances_
features = X_train_scaled.columns
indices = np.argsort(importances)[::-1]
top_n = 10
top_features = [features[i] for i in indices[:top_n]]
top_importances = importances[indices[:top_n]]

print("Top Feature Importances:")
for name, score in zip(top_features, top_importances):
    print(f"{name}: {score:.4f}")

plt.figure(figsize=(8,5))
plt.barh(top_features[::-1], top_importances[::-1], color='steelblue')
plt.xlabel("Feature Importance")
plt.title("Top 10 Feature Importances (XGBoost)")
plt.tight_layout()
plt.show()

# STEP 9: Amenity Analysis vs. House Price

amenities = [
    'has_basement',
    'poolcnt_clean',    
    'garagecarcnt_clean',
    'fireplacecnt_clean'
]

for amenity in amenities:
    if amenity in df_sample.columns:
        df_sample['has_'+amenity] = (df_sample[amenity].fillna(0) > 0).astype(int)
        group_col = 'has_'+amenity
        group_with = df_sample[df_sample[group_col] == 1]['taxvaluedollarcnt_clean']
        group_without = df_sample[df_sample[group_col] == 0]['taxvaluedollarcnt_clean']

        if len(group_with) == 0 or len(group_without) == 0:
            print(f"\nAmenity: '{amenity}' (Only one group present)")
            if len(group_with) > 0:
                print(f"  All with amenity. Avg price = {group_with.mean():,.0f}")
            elif len(group_without) > 0:
                print(f"  All without amenity. Avg price = {group_without.mean():,.0f}")
            else:
                print("  No data for this amenity.")
        else:
            print(f"\nAmenity: '{amenity}' present vs. absent")
            print(f"  With amenity:    Avg price = {group_with.mean():,.0f}")
            print(f"  Without amenity: Avg price = {group_without.mean():,.0f}")

            plt.figure(figsize=(6, 3))
            plt.boxplot([group_without, group_with], tick_labels=['No', 'Yes'])
            plt.title(f'House Price by {amenity}')
            plt.ylabel('House Price')
            plt.show()
