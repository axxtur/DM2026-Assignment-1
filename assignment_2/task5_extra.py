import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.optimize import linear_sum_assignment
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

# Load Data
df = pd.read_csv('mobile_price.csv')
X = df.drop('price_range', axis=1)
y = df['price_range']
features = X.columns.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def map_clusters_to_labels(y_true, y_pred):
    contingency = pd.crosstab(y_true, y_pred).values
    row_ind, col_ind = linear_sum_assignment(-contingency)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    y_pred_mapped = np.array([mapping[label] for label in y_pred])
    return y_pred_mapped

def evaluate_clustering(y_true, y_pred):
    y_mapped = map_clusters_to_labels(y_true, y_pred)
    acc = accuracy_score(y_true, y_mapped)
    prec = precision_score(y_true, y_mapped, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_mapped, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_mapped, average='weighted', zero_division=0)
    return acc, prec, rec, f1

df_cat = pd.DataFrame()
for col in features:
    bins = np.unique(np.percentile(df[col], [0, 33.3, 66.6, 100]))
    if len(bins) == 4:
        current_labels = ['low', 'medium', 'high']
    elif len(bins) == 3:
        current_labels = ['low', 'high']
    else:
        current_labels = ['unique']
    df_cat[col] = pd.cut(df[col], bins=bins, labels=current_labels, include_lowest=True)
    df_cat[col] = col + '_' + df_cat[col].astype(str)

df_cat['target'] = 'price_' + df['price_range'].astype(str)
transactions = df_cat.values.tolist()
te = TransactionEncoder()
te_ary = te.fit_transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# --- Fast Rule Generation (Optimization) ---
target_labels = frozenset({'price_0', 'price_1', 'price_2', 'price_3'})

def get_target_rules_fast(f_items, targets, min_lift=1.0):
    support_dict = dict(zip(f_items.itemsets, f_items.support))
    rules = []
    for itemset, supp_I in zip(f_items.itemsets, f_items.support):
        targets_in_itemset = itemset.intersection(targets)
        if len(targets_in_itemset) == 1:
            T = frozenset([list(targets_in_itemset)[0]])
            A = itemset - T
            if len(A) > 0 and A in support_dict and T in support_dict:
                lift = supp_I / (support_dict[A] * support_dict[T])
                if lift >= min_lift:
                    rules.append({'antecedents': A, 'consequents': T, 'lift': lift})
    return pd.DataFrame(rules) if rules else pd.DataFrame(columns=['antecedents', 'consequents', 'lift'])

# Sensitivity Analysis Settings
supports = [0.02, 0.05, 0.10, 0.15]
seeds = [0, 10, 42, 100, 999]
sensitivity_results = []
start_total = time.time()

for supp in supports:
    print(f"\n--- Processing Support: {supp} ---")
    start_supp = time.time()
    
    print(f"  Running FP-Growth...")
    f_items = fpgrowth(df_encoded, min_support=supp, use_colnames=True)
    print(f"  Found {len(f_items)} frequent itemsets.")
    
    if len(f_items) == 0: continue
        
    print(f"  Extracting target rules (Optimized)...")
    t_rules = get_target_rules_fast(f_items, target_labels, min_lift=1.0)
    print(f"  Found {len(t_rules)} target-related rules.")
    
    fw = {f: 1.0 for f in features}
    for _, row in t_rules.iterrows():
        lift = row['lift']
        for ant in row['antecedents']:
            feature_name = ant.rsplit('_', 1)[0]
            if feature_name in fw: fw[feature_name] += (lift - 1.0)
    
    avg_w = np.mean(list(fw.values()))
    fw = {k: v / avg_w for k, v in fw.items()}
    
    X_w = X_scaled.copy()
    for i, col in enumerate(features):
        X_w[:, i] = X_w[:, i] * fw[col]
        
    f1_scores, acc_scores = [], []
    print(f"  Running KMeans for {len(seeds)} seeds...")
    for s in seeds:
        kmeans = KMeans(n_clusters=4, random_state=s, n_init=10)
        labels = kmeans.fit_predict(X_w)
        acc, _, _, f1 = evaluate_clustering(y, labels)
        f1_scores.append(f1)
        acc_scores.append(acc)
    
    sensitivity_results.append({
        'min_support': supp, 
        'avg_f1': np.mean(f1_scores), 
        'std_f1': np.std(f1_scores),
        'avg_acc': np.mean(acc_scores)
    })
    print(f"  Done in {time.time() - start_supp:.2f}s")

# --- Final Comparison Table ---
print("\n" + "="*65)
print(f"{'Method':<25} | {'Avg Accuracy':<15} | {'Avg F1-Score':<15}")
print("-" * 65)

# Calculate Baseline
b_accs, b_f1s = [], []
for s in seeds:
    km = KMeans(n_clusters=4, random_state=s, n_init=10)
    l = km.fit_predict(X_scaled)
    acc, _, _, f1 = evaluate_clustering(y, l)
    b_accs.append(acc); b_f1s.append(f1)
print(f"{'Original K-Means':<25} | {np.mean(b_accs):.4f}          | {np.mean(b_f1s):.4f}")

for res in sensitivity_results:
    name = f"Enhanced (Supp {res['min_support']})"
    print(f"{name:<25} | {res['avg_acc']:.4f}          | {res['avg_f1']:.4f}")
print("="*65)

# Plots
df_sens = pd.DataFrame(sensitivity_results)
plt.figure(figsize=(8, 5))
plt.errorbar(df_sens['min_support'], df_sens['avg_f1'], yerr=df_sens['std_f1'], fmt='-o', capsize=5, color='blue')
plt.title('Sensitivity Analysis: F1-Score vs. Min Support')
plt.xlabel('Minimum Support'); plt.ylabel('Average F1-Score'); plt.grid(True, alpha=0.3)
plt.savefig('task5/06_sensitivity.png')

print(f"\nTotal time: {time.time() - start_total:.2f}s. Results saved in task5/")
