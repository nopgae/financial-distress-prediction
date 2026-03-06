"""Generate visualization PNGs for financial-distress-prediction."""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve, auc
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

os.makedirs('images', exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────
df = pd.read_csv('train_data.csv')
y = df['Class']
X = df.drop(columns=['Class', 'Name', 'Sector'], errors='ignore').select_dtypes(include='number')
X = X.fillna(X.median())

# ── Chart 1: Class Distribution ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))

counts = y.value_counts().sort_index()
labels = ['Healthy (0)', 'Distressed (1)']
colors = ['#38A169', '#E53E3E']
bars = ax.bar(labels, counts.values, color=colors, edgecolor='white',
              linewidth=1.5, width=0.5)

for bar, count in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f'{count:,}\n({count/len(y)*100:.1f}%)',
            ha='center', va='bottom', fontsize=13, fontweight='bold')

ax.set_ylabel('Number of Companies', fontsize=13, labelpad=10)
ax.set_title('Class Distribution — Financial Distress Dataset\n(Imbalanced: 72% healthy vs 28% distressed)',
             fontsize=16, fontweight='bold', pad=15)
ax.set_ylim(0, counts.max() * 1.2)
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=10)
ax.spines[['top', 'right']].set_visible(False)
ax.yaxis.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('images/class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print('✓ class_distribution.png')

# ── Chart 2: ROC Curve ────────────────────────────────────────────────────
print('Training XGBoost for ROC curve (cross-val)...')
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)

model = XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric='logloss',
    random_state=42, n_jobs=-1
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_prob = cross_val_predict(model, X_scaled, y_res, cv=cv, method='predict_proba')[:, 1]

fpr, tpr, _ = roc_curve(y_res, y_prob)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(9, 7))
ax.plot(fpr, tpr, color='#2B6CB0', lw=2.5,
        label=f'XGBoost ROC (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random Classifier (AUC = 0.500)')
ax.fill_between(fpr, tpr, alpha=0.08, color='#2B6CB0')

ax.set_xlabel('False Positive Rate', fontsize=13, labelpad=10)
ax.set_ylabel('True Positive Rate', fontsize=13, labelpad=10)
ax.set_title('ROC Curve — XGBoost + SMOTE (5-Fold CV)\nFinancial Distress Prediction',
             fontsize=16, fontweight='bold', pad=15)
ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
ax.tick_params(labelsize=10)
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('images/roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'✓ roc_curve.png  (AUC={roc_auc:.3f})')

# ── Chart 3: Feature Importance ───────────────────────────────────────────
print('Training final model for feature importance...')
model.fit(X_scaled, y_res)
importances = pd.Series(model.feature_importances_, index=X.columns)
top15 = importances.nlargest(15).sort_values()

fig, ax = plt.subplots(figsize=(11, 7))
colors_bar = ['#2B6CB0' if i >= len(top15) - 5 else '#718096' for i in range(len(top15))]
bars = ax.barh(top15.index, top15.values, color=colors_bar,
               edgecolor='white', linewidth=0.8, height=0.7)

for bar, val in zip(bars, top15.values):
    ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', fontsize=10)

ax.set_xlabel('Feature Importance (XGBoost Gain)', fontsize=13, labelpad=10)
ax.set_title('Top 15 Feature Importances — Financial Distress Prediction\n(blue = top 5 features)',
             fontsize=16, fontweight='bold', pad=15)
ax.tick_params(axis='y', labelsize=11)
ax.tick_params(axis='x', labelsize=10)
ax.set_xlim(0, top15.max() * 1.15)
ax.spines[['top', 'right']].set_visible(False)
ax.xaxis.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('images/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print('✓ feature_importance.png')

print('\nAll charts saved to images/')
