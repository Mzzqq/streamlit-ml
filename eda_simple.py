import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=== BANK MARKETING DATASET - EDA ANALYSIS ===\n")

# Load dataset
print("Loading dataset...")
df = pd.read_csv('train.csv', delimiter=';')

print(f"✅ Dataset loaded successfully!")
print(f"📊 Dataset shape: {df.shape}")
print(f"📋 Columns: {list(df.columns)}")

# Basic info
print("\n1. BASIC DATASET INFO")
print("=" * 40)
print(df.info())

# Target distribution
print("\n2. TARGET VARIABLE ANALYSIS")
print("=" * 40)
target_counts = df['y'].value_counts()
target_pct = df['y'].value_counts(normalize=True) * 100

print("Target Distribution:")
for target, count in target_counts.items():
    pct = target_pct[target]
    print(f"  {target}: {count:,} ({pct:.1f}%)")

# Class imbalance check
if target_pct.min() < 20:
    print("⚠️  WARNING: Dataset is imbalanced! Consider using SMOTE or class weights.")

# Missing values
print("\n3. MISSING VALUES CHECK")
print("=" * 40)
missing = df.isnull().sum()
if missing.sum() == 0:
    print("✅ No missing values found!")
else:
    print("Missing values found:")
    print(missing[missing > 0])

# Basic statistics for numerical columns
print("\n4. NUMERICAL FEATURES STATISTICS")
print("=" * 40)
numerical_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
print(df[numerical_cols].describe())

# Categorical features overview
print("\n5. CATEGORICAL FEATURES OVERVIEW")
print("=" * 40)
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

for col in categorical_cols:
    unique_count = df[col].nunique()
    print(f"{col}: {unique_count} unique values")
    print(f"  Top values: {df[col].value_counts().head(3).to_dict()}")

# Success rate by categorical features
print("\n6. SUCCESS RATE BY CATEGORIES")
print("=" * 40)
for col in ['job', 'marital', 'education', 'contact']:
    success_rate = df.groupby(col)['y'].apply(lambda x: (x == 'yes').mean() * 100)
    print(f"\nSuccess rate by {col}:")
    for category, rate in success_rate.sort_values(ascending=False).items():
        print(f"  {category}: {rate:.1f}%")

# Correlation analysis
print("\n7. CORRELATION ANALYSIS")
print("=" * 40)
# Create numerical encoding for correlation
df_corr = df.copy()
for col in categorical_cols + ['y']:
    if col in df_corr.columns:
        df_corr[col] = pd.Categorical(df_corr[col]).codes

correlation_with_target = df_corr.corr()['y'].abs().sort_values(ascending=False)
print("Features most correlated with target:")
for feature, corr in correlation_with_target.items():
    if feature != 'y':
        print(f"  {feature}: {corr:.3f}")

# Key insights
print("\n8. KEY INSIGHTS")
print("=" * 40)
total_customers = len(df)
success_rate = (df['y'] == 'yes').mean() * 100

print(f"📈 Overall success rate: {success_rate:.1f}%")
print(f"📞 Average call duration: {df['duration'].mean():.0f} seconds")
print(f"👥 Total customers: {total_customers:,}")

# Duration analysis
long_calls = df[df['duration'] > df['duration'].median()]
short_calls = df[df['duration'] <= df['duration'].median()]

long_success = (long_calls['y'] == 'yes').mean() * 100
short_success = (short_calls['y'] == 'yes').mean() * 100

print(f"📞 Long calls (>{df['duration'].median():.0f}s) success rate: {long_success:.1f}%")
print(f"📞 Short calls (≤{df['duration'].median():.0f}s) success rate: {short_success:.1f}%")

# Recommendations
print("\n9. BUSINESS RECOMMENDATIONS")
print("=" * 40)
print("💡 Based on the analysis:")
print("  1. Focus on longer call durations - they have higher success rates")
print("  2. Target specific job categories with historically higher success")
print("  3. Consider the timing of campaigns (month effects)")
print("  4. Address class imbalance in modeling")
print("  5. Use previous campaign outcomes as strong predictors")

print("\n✅ EDA Analysis Complete!")
print("Next steps: Data Preparation → Modeling → Deployment") 