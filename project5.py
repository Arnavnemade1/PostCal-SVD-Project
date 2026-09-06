import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds

M = pd.read_csv('co_occur.csv', header=None).values

m_tilde = np.log1p(M)

U, S, Vt = svds(m_tilde, k=100)

np.savez('svd_top100.npz',U=U, S=S, Vt=Vt)

plt.figure(figsize=(8,5))
plt.plot(range(1,101),S, linewidth=2)
plt.title('Top 100 Singular Values')
plt.xlabel('Index $i$')
plt.ylabel('Singular Value Index')
plt.grid(True)
plt.tight_layout()
plt.savefig('singular_values_plot.png')
plt.show()
