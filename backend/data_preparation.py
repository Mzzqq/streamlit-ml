import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib

print("=== DATA PREPARATION FOR BANK MARKETING ===\n")

def load_data():
    """Load dan prepare dataset"""
    print("📥 Loading dataset...")
    df = pd.read_csv('train.csv', delimiter=';')
    print(f"✅ Dataset loaded: {df.shape}")
    return df

def feature_engineering(df):
    """Feature engineering dan transformasi"""
    print("\n🛠️  FEATURE ENGINEERING")
    print("=" * 40)
    
    df_processed = df.copy()
    
    # 1. Handle pdays (karena -1 berarti tidak pernah dihubungi)
    df_processed['was_contacted_before'] = (df_processed['pdays'] != -1).astype(int)
    df_processed['pdays_clean'] = df_processed['pdays'].replace(-1, 0)
    
    # 2. Create age groups
    df_processed['age_group'] = pd.cut(df_processed['age'], 
                                     bins=[0, 30, 40, 50, 60, 100], 
                                     labels=['18-30', '31-40', '41-50', '51-60', '60+'])
    
    # 3. Create balance categories
    df_processed['balance_category'] = pd.cut(df_processed['balance'], 
                                            bins=[-np.inf, 0, 1000, 5000, np.inf],
                                            labels=['negative', 'low', 'medium', 'high'])
    
    # 4. Duration categories (berdasarkan EDA, ini feature penting)
    df_processed['duration_category'] = pd.cut(df_processed['duration'],
                                             bins=[0, 120, 300, 600, np.inf],
                                             labels=['very_short', 'short', 'medium', 'long'])
    
    # 5. Contact success indicator
    df_processed['contact_cellular'] = (df_processed['contact'] == 'cellular').astype(int)
    
    # 6. Previous campaign success
    df_processed['prev_success'] = (df_processed['poutcome'] == 'success').astype(int)
    
    print(f"✅ Feature engineering complete")
    print(f"📊 New features added: was_contacted_before, age_group, balance_category, duration_category")
    
    return df_processed

def prepare_features(df):
    """Prepare features untuk modeling"""
    print("\n🔧 FEATURE PREPARATION")
    print("=" * 40)
    
    # Define feature sets
    numerical_features = ['age', 'balance', 'duration', 'campaign', 'pdays_clean', 'previous']
    categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
                          'contact', 'month', 'poutcome', 'age_group', 'balance_category', 
                          'duration_category']
    binary_features = ['was_contacted_before', 'contact_cellular', 'prev_success']
    
    # Target variable
    target = 'y'
    
    # Separate features and target
    X = df[numerical_features + categorical_features + binary_features]
    y = df[target]
    
    # Encode target variable
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    print(f"📊 Features selected: {len(X.columns)}")
    print(f"   - Numerical: {len(numerical_features)}")
    print(f"   - Categorical: {len(categorical_features)}")
    print(f"   - Binary: {len(binary_features)}")
    print(f"🎯 Target classes: {le_target.classes_}")
    
    # Create preprocessing pipelines
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            ('bin', 'passthrough', binary_features)
        ]
    )
    
    return X, y_encoded, preprocessor, le_target

def split_and_balance_data(X, y, preprocessor, test_size=0.2, random_state=42):
    """Split data dan handle class imbalance"""
    print("\n⚖️  DATA SPLITTING & BALANCING")
    print("=" * 40)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"📊 Train set: {X_train.shape}")
    print(f"📊 Test set: {X_test.shape}")
    
    # Fit preprocessor and transform data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"📊 Features after preprocessing: {X_train_processed.shape[1]}")
    
    # Check class distribution before balancing
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\n📈 Class distribution before balancing:")
    for class_val, count in zip(unique, counts):
        print(f"   Class {class_val}: {count} ({count/len(y_train)*100:.1f}%)")
    
    # Apply SMOTE for balancing
    smote = SMOTE(random_state=random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)
    
    # Check class distribution after balancing
    unique, counts = np.unique(y_train_balanced, return_counts=True)
    print(f"\n📈 Class distribution after balancing:")
    for class_val, count in zip(unique, counts):
        print(f"   Class {class_val}: {count} ({count/len(y_train_balanced)*100:.1f}%)")
    
    return X_train_balanced, X_test_processed, y_train_balanced, y_test

def save_preprocessing_objects(preprocessor, le_target):
    """Save preprocessing objects untuk deployment"""
    print("\n💾 SAVING PREPROCESSING OBJECTS")
    print("=" * 40)
    
    joblib.dump(preprocessor, 'preprocessor.pkl')
    joblib.dump(le_target, 'label_encoder.pkl')
    
    print("✅ Saved: preprocessor.pkl")
    print("✅ Saved: label_encoder.pkl")

def main():
    """Main function untuk data preparation"""
    # Load data
    df = load_data()
    
    # Feature engineering
    df_processed = feature_engineering(df)
    
    # Prepare features
    X, y, preprocessor, le_target = prepare_features(df_processed)
    
    # Split and balance data
    X_train, X_test, y_train, y_test = split_and_balance_data(X, y, preprocessor)
    
    # Save preprocessing objects
    save_preprocessing_objects(preprocessor, le_target)
    
    # Save processed data
    np.save('X_train_balanced.npy', X_train)
    np.save('X_test_processed.npy', X_test)
    np.save('y_train_balanced.npy', y_train)
    np.save('y_test.npy', y_test)
    
    print("\n✅ DATA PREPARATION COMPLETE!")
    print("📁 Files saved:")
    print("   - X_train_balanced.npy")
    print("   - X_test_processed.npy") 
    print("   - y_train_balanced.npy")
    print("   - y_test.npy")
    print("   - preprocessor.pkl")
    print("   - label_encoder.pkl")
    
    print(f"\n📊 Final dataset info:")
    print(f"   - Training samples: {X_train.shape[0]:,}")
    print(f"   - Test samples: {X_test.shape[0]:,}")
    print(f"   - Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = main() 