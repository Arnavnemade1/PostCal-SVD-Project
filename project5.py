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

def plot_projections(words_list, projs, title, filename):
    plt.figure(figsize=(12, 4))
    plt.axhline(0, color='black', linewidth=1.2)
    sorted_pairs = sorted(zip(projs, words_list), key=lambda x: x[0])
    y_offsets = [0.15, -0.15, 0.3, -0.3, 0.45, -0.45]
    for i, (p, w) in enumerate(sorted_pairs):
        y = y_offsets[i % len(y_offsets)]
        plt.scatter(p, 0, color='red' if p > 0 else 'blue', s=40, zorder=3)
        plt.plot([p, p], [0, y], color='gray', linestyle=':', linewidth=0.8)
        plt.text(p, y + (0.03 if y > 0 else -0.05), f'{w} ({p:.2f})', 
                 ha='center', va='bottom' if y > 0 else 'top', fontsize=10, weight='bold')
    plt.xlim(min(projs) - 0.08, max(projs) + 0.08)
    plt.ylim(-0.7, 0.7)
    plt.xlabel('Projection onto $v = v_{woman} - v_{man}$', fontsize=11)
    plt.yticks([])
    plt.title(title, fontsize=13)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

word2idx = {w: i for i, w in enumerate(dictionary)}
U_norm = U / np.linalg.norm(U, axis=1, keepdims=True)
v = U_norm[word2idx['woman']] - U_norm[word2idx['man']]

words_di = ['boy', 'girl', 'brother', 'sister', 'king', 'queen', 'he', 'she', 'john', 'mary', 'wall', 'tree']
projs_di = [np.dot(U_norm[word2idx[w]], v) for w in words_di]
print("Part d(i) Projections:")
for w, p in zip(words_di, projs_di):
    print(f"{w}: {p:.4f}")
print()
plot_projections(words_di, projs_di, 'Projections onto $v = v_{woman} - v_{man}$ (Part d.i)', 'part1_d_i.png')

words_dii = ['math', 'matrix', 'history', 'nurse', 'doctor', 'pilot', 'teacher', 'engineer', 'science', 'arts', 'literature', 'bob', 'alice']
projs_dii = [np.dot(U_norm[word2idx[w]], v) for w in words_dii]
print("Part d(ii) Projections:")
for w, p in zip(words_dii, projs_dii):
    print(f"{w}: {p:.4f}")
print()
plot_projections(words_dii, projs_dii, 'Projections onto $v = v_{woman} - v_{man}$ (Part d.ii)', 'part1_d_ii.png')

sims_stanford = np.dot(U_norm, U_norm[word2idx['stanford']])
top_stanford = np.argsort(sims_stanford)[::-1]
print("Part f(i) Most similar words to stanford:")
for idx_word in top_stanford[1:11]:
    print(f"{dictionary[idx_word]}: {sims_stanford[idx_word]:.4f}")
print()

analogies = []
with open('analogy_task.txt') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 4:
            analogies.append([word2idx[p] for p in parts])
analogies = np.array(analogies)

w1, w2, w3, w4 = analogies[:, 0], analogies[:, 1], analogies[:, 2], analogies[:, 3]
target = U_norm[w2] - U_norm[w1] + U_norm[w3]
sims = np.dot(target, U_norm.T)

row_idx = np.arange(len(analogies))
sims[row_idx, w1] = -np.inf
sims[row_idx, w2] = -np.inf
sims[row_idx, w3] = -np.inf

top1 = np.argmax(sims, axis=1)
acc_svd = np.mean(top1 == w4)
print(f"Part f(ii) SVD Analogy Accuracy: {acc_svd * 100:.2f}%")
print()

difficult_count = 0
print("Part f(ii) Difficult analogies not in top 5 guesses:")
for i in range(len(analogies)):
    ranked = np.argsort(sims[i])[::-1]
    rank = np.where(ranked == w4[i])[0][0]
    if rank >= 5:
        a_words = [str(dictionary[analogies[i, j]]) for j in range(4)]
        top5_pred = [str(dictionary[ranked[k]]) for k in range(5)]
        print(f"Analogy: {a_words[0]} is to {a_words[1]} as {a_words[2]} is to {a_words[3]}")
        print(f"Correct word rank: {rank + 1}, Top 5 predictions: {top5_pred}")
        print()
        difficult_count += 1
        if difficult_count == 3:
            break

M = pd.read_csv('co_occur.csv', header=None).values
m_tilde = np.log1p(M)
norms_m = np.linalg.norm(m_tilde, axis=1, keepdims=True)
norms_m[norms_m == 0] = 1.0
M_norm = m_tilde / norms_m

target_m = M_norm[w2] - M_norm[w1] + M_norm[w3]
correct_m = 0
for start in range(0, len(analogies), 1000):
    end = min(start + 1000, len(analogies))
    sub_sims = np.dot(target_m[start:end], M_norm.T)
    for r_i in range(end - start):
        a_idx = start + r_i
        sub_sims[r_i, w1[a_idx]] = -np.inf
        sub_sims[r_i, w2[a_idx]] = -np.inf
        sub_sims[r_i, w3[a_idx]] = -np.inf
    correct_m += np.sum(np.argmax(sub_sims, axis=1) == w4[start:end])

acc_m = correct_m / len(analogies)
print(f"Part f(iii) Normalized M_tilde Analogy Accuracy: {acc_m * 100:.2f}%")
print()

W_bonus = U * (S ** 0.25)
W_bonus_norm = W_bonus / np.linalg.norm(W_bonus, axis=1, keepdims=True)
target_bonus = W_bonus_norm[w2] - W_bonus_norm[w1] + W_bonus_norm[w3]
sims_bonus = np.dot(target_bonus, W_bonus_norm.T)
sims_bonus[row_idx, w1] = -np.inf
sims_bonus[row_idx, w2] = -np.inf
sims_bonus[row_idx, w3] = -np.inf
acc_bonus = np.mean(np.argmax(sims_bonus, axis=1) == w4)
print(f"Part f(v) Bonus Weighted SVD (alpha=0.25) Analogy Accuracy: {acc_bonus * 100:.2f}%")
