import numpy as np

# Mapping for one-hot encoding
BASE_TO_VECTOR = {
    'A': np.array([1, 0, 0, 0]),
    'C': np.array([0, 1, 0, 0]),
    'G': np.array([0, 0, 1, 0]),
    'T': np.array([0, 0, 0, 1]),
    'U': np.array([0, 0, 0, 1])  # Treats U the same as T numerically
}

# Converts a DNA/RNA sequence string into a flattened one-hot encoded vector
def one_hot_encode(sequence):
    vectors = [BASE_TO_VECTOR[base] for base in sequence]
    matrix = np.column_stack(vectors)      # 4 x n matrix
    flat_vector = matrix.flatten()         # length = 4n
    return flat_vector

def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def detect_sequence_type(seq):
    bases = set(seq)
    has_t = 'T' in bases
    has_u = 'U' in bases

    if has_t and has_u:
        raise ValueError("Invalid sequence: cannot contain both T and U.")
    elif has_u:
        return "RNA"
    else:
        return "DNA"

def validate_length(seq):
    if not (30 <= len(seq) <= 50):
        raise ValueError("Sequence length must be between 30 and 50 bases.")

def compare_sequences(seq1, seq2):
    validate_length(seq1)
    validate_length(seq2)

    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be the same length.")

    type1 = detect_sequence_type(seq1)
    type2 = detect_sequence_type(seq2)

    if type1 != type2:
        raise ValueError(f"Sequence type mismatch: {type1} vs {type2}.")

    v1 = one_hot_encode(seq1)
    v2 = one_hot_encode(seq2)

    dist = euclidean_distance(v1, v2)
    cos_sim = cosine_similarity(v1, v2)

    return dist, cos_sim

def main():
    while True:
        try:
            seq1 = input("Enter first sequence: ").upper()
            seq2 = input("Enter second sequence: ").upper()

            dist, sim = compare_sequences(seq1, seq2)

            print("\nEuclidean distance:", dist)
            print("Cosine similarity:", sim)
            break  # Exit loop if everything succeeds

        except ValueError as e:
            print("\nError:", e)
            print("Please re-enter both sequences.\n")

if __name__ == "__main__":
    main()