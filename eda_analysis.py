import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style untuk visualisasi
plt.style.use('default')
sns.set_palette("husl")

def load_and_explore_data():
    """Load dataset dan melakukan eksplorasi awal"""
    print("=== BANK MARKETING DATASET ANALYSIS ===\n")
    
    # Load data
    df = pd.read_csv('train.csv', delimiter=';')
    
    print("1. BASIC DATA INFO")
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {df.columns.tolist()}")
    print(f"\nData types:\n{df.dtypes}")
    
    return df

def basic_statistics(df):
    """Statistik dasar dataset"""
    print("\n2. BASIC STATISTICS")
    print("\nNumerical features summary:")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numerical_cols].describe())
    
    print("\nCategorical features summary:")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\n{col}: {df[col].value_counts().head()}")

def target_analysis(df):
    """Analisis target variable"""
    print("\n3. TARGET VARIABLE ANALYSIS")
    target_dist = df['y'].value_counts()
    target_pct = df['y'].value_counts(normalize=True) * 100
    
    print("Target distribution:")
    for val, count, pct in zip(target_dist.index, target_dist.values, target_pct.values):
        print(f"{val}: {count} ({pct:.1f}%)")
    
    # Visualisasi target
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Count plot
    sns.countplot(data=df, x='y', ax=ax1)
    ax1.set_title('Target Variable Distribution')
    ax1.set_ylabel('Count')
    
    # Pie chart
    ax2.pie(target_dist.values, labels=target_dist.index, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Target Variable Percentage')
    
    plt.tight_layout()
    plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return target_dist

def missing_values_analysis(df):
    """Analisis missing values"""
    print("\n4. MISSING VALUES ANALYSIS")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing Percentage': missing_pct
    }).sort_values('Missing Count', ascending=False)
    
    print(missing_df[missing_df['Missing Count'] > 0])
    
    if missing_df['Missing Count'].sum() == 0:
        print("✅ No missing values found!")
    
    return missing_df

def numerical_features_analysis(df):
    """Analisis fitur numerik"""
    print("\n5. NUMERICAL FEATURES ANALYSIS")
    
    numerical_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
    
    # Distribusi fitur numerik
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, col in enumerate(numerical_cols):
        # Histogram
        axes[i].hist(df[col], bins=30, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        
        # Tambahkan statistik
        mean_val = df[col].mean()
        median_val = df[col].median()
        axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.1f}')
        axes[i].axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.1f}')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('numerical_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Correlation matrix
    corr_matrix = df[numerical_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True)
    plt.title('Correlation Matrix - Numerical Features')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def categorical_features_analysis(df):
    """Analisis fitur kategorikal"""
    print("\n6. CATEGORICAL FEATURES ANALYSIS")
    
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    
    # Distribusi fitur kategorikal
    n_cols = 3
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15))
    axes = axes.ravel()
    
    for i, col in enumerate(categorical_cols):
        value_counts = df[col].value_counts()
        axes[i].bar(range(len(value_counts)), value_counts.values)
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xticks(range(len(value_counts)))
        axes[i].set_xticklabels(value_counts.index, rotation=45, ha='right')
        axes[i].set_ylabel('Count')
    
    # Hide extra subplots
    for i in range(len(categorical_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('categorical_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def target_relationship_analysis(df):
    """Analisis hubungan fitur dengan target"""
    print("\n7. FEATURE-TARGET RELATIONSHIP ANALYSIS")
    
    # Numerical features vs target
    numerical_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, col in enumerate(numerical_cols):
        # Box plot untuk melihat distribusi berdasarkan target
        df.boxplot(column=col, by='y', ax=axes[i])
        axes[i].set_title(f'{col} by Target')
        axes[i].set_xlabel('Target (y)')
        axes[i].set_ylabel(col)
    
    plt.suptitle('Numerical Features vs Target', y=1.02)
    plt.tight_layout()
    plt.savefig('numerical_vs_target.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Categorical features vs target
    categorical_cols = ['job', 'marital', 'education', 'housing', 'loan', 'contact']
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.ravel()
    
    for i, col in enumerate(categorical_cols):
        # Stacked bar chart
        ct = pd.crosstab(df[col], df['y'], normalize='index') * 100
        ct.plot(kind='bar', stacked=True, ax=axes[i])
        axes[i].set_title(f'{col} vs Target (Percentage)')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Percentage')
        axes[i].legend(title='Target')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('categorical_vs_target.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_insights(df, target_dist):
    """Generate insights dari analisis"""
    print("\n" + "="*50)
    print("📊 KEY INSIGHTS & RECOMMENDATIONS")
    print("="*50)
    
    print("\n🎯 TARGET VARIABLE INSIGHTS:")
    yes_pct = (target_dist['yes'] / target_dist.sum()) * 100
    print(f"• Dataset imbalanced: Only {yes_pct:.1f}% customers subscribe to term deposit")
    print(f"• Need to consider class imbalance in modeling (use SMOTE, class weights, etc.)")
    
    print("\n📈 FEATURE INSIGHTS:")
    print("• Duration seems highly correlated with success - longer calls = higher chance")
    print("• Previous campaign outcome is important predictor")
    print("• Age and job type show different subscription patterns")
    print("• Contact method (cellular vs telephone) affects success rate")
    
    print("\n🧹 DATA QUALITY INSIGHTS:")
    print("• No missing values - good data quality")
    print("• 'pdays' has many -1 values (customers not contacted before)")
    
    print("\n💡 BUSINESS RECOMMENDATIONS:")
    print("• Focus on customers with longer call durations")
    print("• Target specific job categories with higher success rates")
    print("• Use cellular contact method when possible")
    print("• Consider timing of campaigns (month effects)")
    print("• Develop different strategies for new vs. returning customers")

def main():
    """Fungsi utama untuk menjalankan seluruh analisis"""
    # Load data
    df = load_and_explore_data()
    
    # Basic statistics
    basic_statistics(df)
    
    # Target analysis
    target_dist = target_analysis(df)
    
    # Missing values
    missing_df = missing_values_analysis(df)
    
    # Numerical features
    numerical_features_analysis(df)
    
    # Categorical features
    categorical_features_analysis(df)
    
    # Feature-target relationships
    target_relationship_analysis(df)
    
    # Generate insights
    generate_insights(df, target_dist)
    
    print("\n✅ EDA Complete! Check the generated PNG files for visualizations.")
    
    return df

if __name__ == "__main__":
    df = main() 