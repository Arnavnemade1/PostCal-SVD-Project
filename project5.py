import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds

if os.path.exists('svd_top100.npz'):
    data = np.load('svd_top100.npz')
    U, S, Vt = data['U'], data['S'], data['Vt']
else:
    M = pd.read_csv('co_occur.csv', header=None).values
    m_tilde = np.log1p(M)
    U, S, Vt = svds(m_tilde, k=100)

idx = np.argsort(S)[::-1]
S = S[idx]
U = U[:, idx]
Vt = Vt[idx, :]

np.savez('svd_top100.npz', U=U, S=S, Vt=Vt)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 101), S, linewidth=2)
plt.title('Top 100 Singular Values')
plt.xlabel('Index $i$')
plt.ylabel('Singular Value $\sigma_i$')
plt.grid(True)
plt.tight_layout()
plt.savefig('singular_values_plot.png')
plt.close()

with open('dictionary.txt') as f:
    dictionary = np.array([line.strip() for line in f])

chosen_indices = [2, 3, 4, 5, 7]
for idx_i in chosen_indices:
    vec = U[:, idx_i]
    top_pos_idx = np.argsort(vec)[-10:][::-1]
    top_neg_idx = np.argsort(vec)[:10]
    print(f"Vector {idx_i + 1} (Singular Value: {S[idx_i]:.2f})")
    print("Top 10 Positive Words:", dictionary[top_pos_idx].tolist())
    print("Top 10 Negative Words:", dictionary[top_neg_idx].tolist())
    print()
