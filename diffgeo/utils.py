def christoffel_terms(Gamma):
    dim = len(Gamma)
    return {
        (k, i, j): Gamma[k][i][j]
        for k in range(dim)
        for i in range(dim)
        for j in range(dim)
        if Gamma[k][i][j] != 0
    }


def format_christoffel(terms):
    return [
        f"Gamma^{k}_{i}{j} = {val}"
        for (k, i, j), val in terms.items()
    ]


def print_lines(lines):
    for line in lines:
        print(line)
        