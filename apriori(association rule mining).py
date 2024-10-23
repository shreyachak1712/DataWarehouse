import csv
from itertools import combinations


# Read transactions from a CSV file
def read_transactions_from_csv(transactions):
   transactions = []
   with open('/Users/Sanjay/Downloads/transactions.csv', mode='r') as file:
       reader = csv.DictReader(file)
       for row in reader:
           # Split the string of items into a list
           transaction = row['items'].split(', ')
           transactions.append(transaction)
   return transactions


# Step 1: Generate frequent 1-itemsets
def find_frequent_1_itemsets(transactions, min_sup):
   item_count = {}
   for transaction in transactions:
       for item in transaction:
           item_count[item] = item_count.get(item, 0) + 1
   return {frozenset([item]): count for item, count in item_count.items() if count >= min_sup}


# Step 2: Generate candidates
def apriori_gen(Lk_1):
   Lk_1_list = list(Lk_1)
   return [Lk_1_list[i] | Lk_1_list[j] for i in range(len(Lk_1_list)) for j in range(i + 1, len(Lk_1_list)) if list(Lk_1_list[i])[:-1] == list(Lk_1_list[j])[:-1]]


# Step 3: Prune candidates
def has_infrequent_subset(candidate, Lk_1):
   return any(frozenset(subset) not in Lk_1 for subset in combinations(candidate, len(candidate) - 1))


# Step 4: Count occurrences of candidates
def count_candidates(transactions, candidates, min_sup):
   candidate_count = {candidate: 0 for candidate in candidates}
   for transaction in transactions:
       transaction_set = set(transaction)
       for candidate in candidates:
           if candidate.issubset(transaction_set):
               candidate_count[candidate] += 1
   return {candidate: count for candidate, count in candidate_count.items() if count >= min_sup}


# Step 5: Apriori Algorithm
def apriori(transactions, min_sup):
   L, k = [find_frequent_1_itemsets(transactions, min_sup)], 2
   while True:
       Ck = [c for c in apriori_gen(L[k - 2].keys()) if not has_infrequent_subset(c, L[k - 2].keys())]
       Lk = count_candidates(transactions, Ck, min_sup)
       if not Lk: break
       L.append(Lk)
       k += 1
   return L


# Step 6: Generate association rules
def generate_association_rules_last_freq(frequent_itemsets, transactions, min_conf):
   rules, transaction_count = [], len(transactions)
   for itemset, support_count in frequent_itemsets[-1].items():
       for antecedent_len in range(1, len(itemset)):
           for antecedents in combinations(itemset, antecedent_len):
               antecedents, consequents = frozenset(antecedents), itemset - frozenset(antecedents)
               confidence = support_count / get_support(antecedents, frequent_itemsets)
               if confidence >= min_conf:
                   rules.append((antecedents, consequents, support_count / transaction_count, confidence))
   return rules


# Helper function to get support
def get_support(itemset, frequent_itemsets):
   for k_itemsets in frequent_itemsets:
       if itemset in k_itemsets:
           return k_itemsets[itemset]
   return 0


# Read transactions from CSV
transactions = read_transactions_from_csv('transactions.csv')


# Run Apriori Algorithm
frequent_itemsets = apriori(transactions, 2)


# Display the frequent itemsets
for i, itemsets in enumerate(frequent_itemsets):
   print(f"Frequent {i + 1}-itemsets:")
   for itemset, count in itemsets.items():
       print(f"  {set(itemset)}: {count}")


# Generate and print association rules
rules = generate_association_rules_last_freq(frequent_itemsets, transactions, 0)
print("\nAssociation Rules from Last Frequent Itemset:")
for antecedents, consequents, support, confidence in rules:
   print(f"Rule: {set(antecedents)} => {set(consequents)} (Support: {support:.2f}, Confidence: {confidence:.2f})")
