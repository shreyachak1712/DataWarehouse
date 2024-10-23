import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

file_path = 'games_sales_dataset.csv'
df = pd.read_csv('/Users/Sanjay/Downloads/games_sales_dataset.csv')

df_cleaned = df.dropna(axis=1, how='all')

print("Cleaned Dataset Preview:")
print(df_cleaned.head())

basket = df_cleaned.apply(lambda row: row.dropna().tolist(), axis=1).tolist()

te = TransactionEncoder()
te_ary = te.fit(basket).transform(basket)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

min_support = 0.05
frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)

print("\nFrequent Itemsets:")
print(frequent_itemsets)

min_confidence = 0.6  
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
