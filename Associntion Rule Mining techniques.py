import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

# Sample transaction dataset
transactions = [
    ['Milk', 'Bread', 'Butter'],
    ['Beer', 'Bread'],
    ['Milk', 'Bread', 'Beer', 'Butter'],
    ['Bread', 'Butter'],
    ['Milk', 'Butter'],
    ['Beer', 'Bread', 'Butter']
]

# Convert transactions into one-hot encoded DataFrame
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)

# -----------------------------
# Apriori Algorithm
# -----------------------------
frequent_itemsets_apriori = apriori(df, min_support=0.3, use_colnames=True)

rules_apriori = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=0.6)

print("Apriori Frequent Itemsets:\n", frequent_itemsets_apriori)
print("\nApriori Association Rules:\n", rules_apriori)

# -----------------------------
# FP-Growth Algorithm
# -----------------------------
frequent_itemsets_fp = fpgrowth(df, min_support=0.3, use_colnames=True)

rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=0.6)

print("\nFP-Growth Frequent Itemsets:\n", frequent_itemsets_fp)
print("\nFP-Growth Association Rules:\n", rules_fp)
