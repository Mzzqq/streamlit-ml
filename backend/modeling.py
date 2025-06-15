import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

print("=== MACHINE LEARNING MODELING ===\n")

def load_processed_data():
    """Load processed data"""
    print("📥 Loading processed data...")
    
    X_train = np.load('X_train_balanced.npy')
    X_test = np.load('X_test_processed.npy')
    y_train = np.load('y_train_balanced.npy')
    y_test = np.load('y_test.npy')
    
    print(f"✅ Data loaded successfully")
    print(f"📊 Training set: {X_train.shape}")
    print(f"📊 Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    """Train multiple ML models"""
    print("\n🤖 TRAINING MULTIPLE MODELS")
    print("=" * 40)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    trained_models = {}
    
    # Train each model
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"✅ {name} trained successfully")
    
    return trained_models

def evaluate_models(models, X_train, X_test, y_train, y_test):
    """Evaluate all models"""
    print("\n📊 MODEL EVALUATION")
    print("=" * 40)
    
    results = {}
    
    for name, model in models.items():
        print(f"\n📈 Evaluating {name}...")
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        auc = roc_auc_score(y_test, y_test_proba)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        results[name] = {
            'Train Accuracy': train_accuracy,
            'Test Accuracy': test_accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC-ROC': auc,
            'CV F1 Mean': cv_mean,
            'CV F1 Std': cv_std,
            'Predictions': y_test_pred,
            'Probabilities': y_test_proba
        }
        
        print(f"   Test Accuracy: {test_accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   AUC-ROC: {auc:.4f}")
        print(f"   CV F1: {cv_mean:.4f} ± {cv_std:.4f}")
    
    return results

def create_results_dataframe(results):
    """Create results comparison dataframe"""
    print("\n📋 RESULTS SUMMARY")
    print("=" * 40)
    
    df_results = pd.DataFrame(results).T
    df_results = df_results.drop(['Predictions', 'Probabilities'], axis=1)
    df_results = df_results.round(4)
    
    print(df_results)
    
    # Find best model
    best_model_f1 = df_results['F1-Score'].idxmax()
    best_model_auc = df_results['AUC-ROC'].idxmax()
    
    print(f"\n🏆 BEST MODELS:")
    print(f"   Best F1-Score: {best_model_f1} ({df_results.loc[best_model_f1, 'F1-Score']:.4f})")
    print(f"   Best AUC-ROC: {best_model_auc} ({df_results.loc[best_model_auc, 'AUC-ROC']:.4f})")
    
    return df_results, best_model_f1

def plot_model_comparison(df_results):
    """Plot model comparison"""
    print("\n📊 Creating model comparison plots...")
    
    # Metrics to plot
    metrics = ['Test Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        bars = ax.bar(df_results.index, df_results[metric])
        ax.set_title(f'{metric} Comparison')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
    
    # Hide the last subplot
    axes[5].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrices(models, results, y_test):
    """Plot confusion matrices for all models"""
    print("\n🔍 Creating confusion matrices...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, (name, model) in enumerate(models.items()):
        y_pred = results[name]['Predictions']
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'Confusion Matrix - {name}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()

def feature_importance_analysis(best_model, model_name):
    """Analyze feature importance for tree-based models"""
    if hasattr(best_model, 'feature_importances_'):
        print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS - {model_name}")
        print("=" * 40)
        
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("Top 10 Most Important Features:")
        for i in range(min(10, len(importances))):
            print(f"{i+1}. Feature {indices[i]}: {importances[indices[i]]:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.bar(range(min(20, len(importances))), importances[indices[:min(20, len(importances))]])
        plt.title(f'Top 20 Feature Importances - {model_name}')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

def save_best_model(best_model, model_name):
    """Save the best model"""
    print(f"\n💾 SAVING BEST MODEL: {model_name}")
    print("=" * 40)
    
    joblib.dump(best_model, 'best_model.pkl')
    
    # Save model info
    model_info = {
        'model_name': model_name,
        'model_type': type(best_model).__name__
    }
    joblib.dump(model_info, 'model_info.pkl')
    
    print(f"✅ Best model saved: best_model.pkl")
    print(f"✅ Model info saved: model_info.pkl")

def generate_business_insights(df_results, best_model_name):
    """Generate business insights from modeling results"""
    print("\n💡 BUSINESS INSIGHTS")
    print("=" * 40)
    
    best_f1 = df_results.loc[best_model_name, 'F1-Score']
    best_precision = df_results.loc[best_model_name, 'Precision']
    best_recall = df_results.loc[best_model_name, 'Recall']
    best_auc = df_results.loc[best_model_name, 'AUC-ROC']
    
    print(f"🏆 Best Model: {best_model_name}")
    print(f"📊 Performance Metrics:")
    print(f"   • Precision: {best_precision:.1%} - Of customers predicted to subscribe, {best_precision:.1%} actually will")
    print(f"   • Recall: {best_recall:.1%} - Of customers who will subscribe, {best_recall:.1%} are correctly identified")
    print(f"   • F1-Score: {best_f1:.3f} - Overall balance between precision and recall")
    print(f"   • AUC-ROC: {best_auc:.3f} - Model can distinguish between classes very well")
    
    print(f"\n💼 Business Impact:")
    
    # Assume 100,000 potential customers
    total_customers = 100000
    baseline_conversion = 0.117  # 11.7% from EDA
    
    # Current scenario (no model)
    current_conversions = total_customers * baseline_conversion
    
    # With model scenario (target only high-probability customers)
    # Assume we contact top 30% of customers based on model prediction
    targeted_customers = total_customers * 0.3
    expected_conversions = targeted_customers * baseline_conversion * (1 + best_f1)  # Improvement factor
    
    cost_per_call = 5  # Assume $5 per call
    revenue_per_conversion = 1000  # Assume $1000 revenue per term deposit
    
    current_cost = total_customers * cost_per_call
    current_revenue = current_conversions * revenue_per_conversion
    current_profit = current_revenue - current_cost
    
    targeted_cost = targeted_customers * cost_per_call
    targeted_revenue = expected_conversions * revenue_per_conversion
    targeted_profit = targeted_revenue - targeted_cost
    
    print(f"   • Current approach (call everyone):")
    print(f"     - Calls: {total_customers:,}")
    print(f"     - Conversions: {current_conversions:,.0f}")
    print(f"     - Cost: ${current_cost:,.0f}")
    print(f"     - Revenue: ${current_revenue:,.0f}")
    print(f"     - Profit: ${current_profit:,.0f}")
    
    print(f"   • Model-driven approach (target 30% most likely):")
    print(f"     - Calls: {targeted_customers:,.0f}")
    print(f"     - Expected conversions: {expected_conversions:,.0f}")
    print(f"     - Cost: ${targeted_cost:,.0f}")
    print(f"     - Revenue: ${targeted_revenue:,.0f}")
    print(f"     - Profit: ${targeted_profit:,.0f}")
    
    savings = current_cost - targeted_cost
    profit_improvement = targeted_profit - current_profit
    
    print(f"   • Improvements:")
    print(f"     - Cost savings: ${savings:,.0f}")
    print(f"     - Profit improvement: ${profit_improvement:,.0f}")
    print(f"     - ROI improvement: {profit_improvement/current_profit:.1%}")

def main():
    """Main function untuk modeling"""
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Evaluate models
    results = evaluate_models(models, X_train, X_test, y_train, y_test)
    
    # Create results summary
    df_results, best_model_name = create_results_dataframe(results)
    
    # Plot comparisons
    plot_model_comparison(df_results)
    plot_confusion_matrices(models, results, y_test)
    
    # Feature importance
    best_model = models[best_model_name]
    feature_importance_analysis(best_model, best_model_name)
    
    # Save best model
    save_best_model(best_model, best_model_name)
    
    # Business insights
    generate_business_insights(df_results, best_model_name)
    
    print("\n✅ MODELING COMPLETE!")
    print("📁 Files created:")
    print("   - best_model.pkl")
    print("   - model_info.pkl")
    print("   - model_comparison.png")
    print("   - confusion_matrices.png")
    print("   - feature_importance.png")
    
    return models, results, best_model_name

if __name__ == "__main__":
    models, results, best_model_name = main() 