def lcs_sequence(X: str, Y: str) -> str:
    X, Y = X.split(), Y.split()
    m, n = len(X), len(Y)
    # Build LCS length table
    L = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            if X[i] == Y[j]:
                L[i + 1][j + 1] = L[i][j] + 1
            else:
                L[i + 1][j + 1] = max(L[i][j + 1], L[i + 1][j])

    # Backtrack to find the LCS sequence
    i, j = m, n
    lcs_words = []
    while i > 0 and j > 0:
        if X[i - 1] == Y[j - 1]:
            lcs_words.append(X[i - 1])
            i -= 1
            j -= 1
        elif L[i - 1][j] >= L[i][j - 1]:
            i -= 1
        else:
            j -= 1

    lcs_words = lcs_words[::-1]  # Reverse the list to correct order
    return " ".join(lcs_words)
