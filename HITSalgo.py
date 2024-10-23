import numpy as np
import scipy.sparse as sp

def hits(M, tol=1e-6, max_iter=100):
    """
    Compute HITS algorithm for hubs and authorities using sparse matrices.
    """
    n = M.shape[0]
    
    # Initialize hub and authority scores
    h = np.ones(n)
    a = np.ones(n)
    
    M = sp.csc_matrix(M, dtype=np.float32)
    
    for i in range(max_iter):
        h_old = h.copy()
        a_old = a.copy()
        
        # Update authority and hub scores
        a = M.T.dot(h_old)
        h = M.dot(a_old)
        
        # Normalize authority scores
        a_norm = np.linalg.norm(a, ord=2)
        if a_norm > 0:
            a /= a_norm
        else:
            a.fill(0)  # If the norm is zero, fill with zeros to avoid issues
        
        # Normalize hub scores
        h_norm = np.linalg.norm(h, ord=2)
        if h_norm > 0:
            h /= h_norm
        else:
            h.fill(0)  # If the norm is zero, fill with zeros to avoid issues
        
        # Check for convergence
        if np.linalg.norm(h - h_old, ord=1) < tol and np.linalg.norm(a - a_old, ord=1) < tol:
            break
    
    return h, a

# Load the Cora dataset
def load_cora_data(path):
    data = np.loadtxt('/Users/Sanjay/Downloads/cora.cites', dtype=int)
    row, col = data[:, 0], data[:, 1]
    num_nodes = max(max(row), max(col)) + 1
    M = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    return M

# Path to the dataset
cora_cites_path = 'cora.cites'  # Adjust this path to your dataset
M = load_cora_data(cora_cites_path)

# Run HITS on the Cora dataset
hub_scores, authority_scores = hits(M)

# Enhanced Output for HITS
top_hub_indices = np.argsort(-hub_scores)[:10]
top_hub_scores = hub_scores[top_hub_indices]

top_auth_indices = np.argsort(-authority_scores)[:10]
top_auth_scores = authority_scores[top_auth_indices]

print("\nTop 10 Hubs:")
print("{:<10} {:<25}".format("Hub ID", "Hub Score"))
print("=" * 40)
for i in range(10):
    print("{:<10} {:<25.15f}".format(top_hub_indices[i], top_hub_scores[i]))

print("\nTop 10 Authorities:")
print("{:<10} {:<25}".format("Authority ID", "Authority Score"))
print("=" * 45)
for i in range(10):
    print("{:<10} {:<25.15f}".format(top_auth_indices[i], top_auth_scores[i]))


