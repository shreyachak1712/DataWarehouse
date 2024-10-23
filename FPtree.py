import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'games_sales_dataset.csv'
df = pd.read_csv('/Users/Sanjay/Downloads/games_sales_dataset.csv')

# Clean dataset by removing columns with all NaN values
df_cleaned = df.dropna(axis=1, how='all')

print("Cleaned Dataset Preview:")
print(df_cleaned.head())

# Convert the dataset into a list of transactions (baskets)
basket = df_cleaned.apply(lambda row: row.dropna().tolist(), axis=1).tolist()

# Encode transactions into one-hot encoding
te = TransactionEncoder()
te_ary = te.fit(basket).transform(basket)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Apply FP-Growth algorithm
min_support = 0.05
frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)

print("\nFrequent Itemsets:")
print(frequent_itemsets)

# Generate association rules
min_confidence = 0.6  
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# --- Tree Visualization ---

# Create a directed graph for item relationships
G = nx.DiGraph()

# Build a tree structure from frequent itemsets
for itemset in frequent_itemsets['itemsets']:
    items = list(itemset)
    # Add edges between items to represent relationships (in this case, we're just chaining them)
    for i in range(len(items) - 1):
        G.add_edge(items[i], items[i + 1])

# Plot the tree using networkx
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)  # Layout for visualization
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=12, font_weight="bold", arrows=True)
plt.title("Frequent Itemsets Relationship Tree")
plt.show()
