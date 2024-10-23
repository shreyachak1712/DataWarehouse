import numpy as np
import scipy.sparse as sp

def pagerank(M, alpha=0.85, tol=1e-6, max_iter=100):
    """
    Compute PageRank of a graph using the power iteration method.
    """
    n = M.shape[0]
    M = M / M.sum(axis=0).A1  # Normalize column-wise
    r = np.ones(n) / n  # Initial PageRank scores
    
    for i in range(max_iter):
        r_old = r.copy()
        r = alpha * M.dot(r_old) + (1 - alpha) / n
        
        # Check for convergence
        if np.linalg.norm(r - r_old, ord=1) < tol:
            break
    
    return r

# Load the Web-Google dataset (adjust the path as necessary)
def load_web_google_data(path):
    data = np.loadtxt('/Users/Sanjay/Downloads/web-Google.txt', dtype=int)
    row, col = data[:, 0], data[:, 1]
    num_nodes = max(max(row), max(col)) + 1
    M = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    return M

# Path to the dataset
web_google_path = 'web-Google.txt'  # Adjust this path to your dataset
M = load_web_google_data(web_google_path)

# Run PageRank on the Web-Google dataset
page_rank_scores = pagerank(M)

# Enhanced Output
top_indices = np.argsort(-page_rank_scores)[:10]
top_scores = page_rank_scores[top_indices]

print("\nTop 10 Pages by PageRank:")
print("{:<10} {:<25} {:<20}".format("Page ID", "PageRank Score", "Link Count"))
print("=" * 55)
for i in range(10):
    link_count = M[:, top_indices[i]].sum()  # Number of outgoing links
    print("{:<10} {:<25.15f} {:<20}".format(top_indices[i], top_scores[i], link_count))

