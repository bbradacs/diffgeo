def christoffel_terms(metric, gamma):
    dim = metric.dim
    return {
        (k, i, j): gamma[k,i,j]
        for k in range(dim)
        for i in range(dim)
        for j in range(dim)
        if gamma[k,i,j] != 0
    }

def riemann_terms(metric,R):
    dim = metric.dim
    return {
        (k, l, i, j): R[k,l,i,j]
        for k in range(dim)
        for l in range(dim)
        for i in range(dim)
        for j in range(dim)
        if R[k,l,i,j] != 0
    }
    
def ricci_terms(metric, ricci):
    dim = metric.dim
    return {
        (i, j): ricci[i,j]
        for i in range(dim)
        for j in range(dim)
        if ricci[i,j] != 0
    }  

def scalar_terms(scalar):
    return scalar

def format_christoffel(terms):
    return [
        f"Gamma^{k}_{i}{j} = {val}"
        for (k, i, j), val in terms.items()
    ]

def format_riemann(terms):
    return [
        f"R^{k}_{l}{i}{j} = {val}"
        for (k, l, i, j), val in terms.items()
    ]

def format_ricci(terms):
    return [
        f"R_{i}{j} = {val}"
        for (i, j), val in terms.items()
    ]

def format_scalar(scalar):
    return f"R = {scalar}"

def print_lines(lines):
    for line in lines:
        print(line)
