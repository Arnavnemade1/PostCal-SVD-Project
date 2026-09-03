import numpy as np
import matplotlib.pyplot as plt
import csv

def load_data():
    """
    Loads the dictionary and co-occurrence matrix.
    """
    print("Loading dictionary...")
    with open('dictionary.txt', 'r') as f:
        dictionary = [line.strip() for line in f.readlines()]
    
    print("Loading co-occurrence matrix (this might take a minute)...")
    # Using numpy to load the large CSV file efficiently
    # M = np.loadtxt('co_occur.csv', delimiter=',')
    # Note: co_occur.csv is quite large. 
    # For a quicker load, you can use pandas:
    import pandas as pd
    M = pd.read_csv('co_occur.csv', header=None).values
    
    return dictionary, M

def part1_b(M):
    """
    Part 1b: Rank-100 approximation of normalized matrix M_tilde.
    """
    print("\n--- Part 1b: Rank-100 Approximation ---")
    # M_tilde_ij = log(1 + M_ij)
    M_tilde = np.log(1 + M)
    
    # Compute SVD. Hint: Try using scipy.sparse.linalg.svds for truncated SVD
    from scipy.sparse.linalg import svds
    # U, S, Vt = svds(M_tilde, k=100)
    
    # TODO: Compute SVD, plot singular values, answer the question.
    pass

def part1_c(U, dictionary):
    """
    Part 1c: Interpret singular vectors.
    """
    print("\n--- Part 1c: Interpret Singular Vectors ---")
    # TODO: Find 5 interesting/interpretable singular vectors and describe semantic/syntactic structures.
    pass

def part1_d(U, dictionary):
    """
    Part 1d: Word embeddings and projections.
    """
    print("\n--- Part 1d: Word Embeddings ---")
    # TODO: Normalize rows of U, project embeddings onto v = v_woman - v_man, and analyze.
    pass

def part1_f(U, dictionary):
    """
    Part 1f: Word similarity and analogy tasks.
    """
    print("\n--- Part 1f: Similarity and Analogy ---")
    # TODO: Define cosine similarity, load analogy_task.txt, evaluate performance.
    pass

def part2_image_compression():
    """
    Part 2: SVD for image compression.
    """
    print("\n--- Part 2: Image Compression ---")
    # Load p5_image.gif (1600x1170)
    # Hint: Use plt.imread() or PIL
    # image = plt.imread('p5_image.gif')
    
    # TODO: Run SVD, recover rank-k approximations, plot, and compute memory.
    pass

if __name__ == '__main__':
    # Uncomment lines below as you progress
    
    # dictionary, M = load_data()
    # U, S, Vt = part1_b(M)
    
    # part1_c(U, dictionary)
    # part1_d(U, dictionary)
    # part1_f(U, dictionary)
    
    part2_image_compression()
