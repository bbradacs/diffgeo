def scalar_curvature(g_up, Ricci):
    dim = g_up.dim

    return sp.simplify(
        sum(
            g_up[i, j] * Ricci[i, j]
            for i in range(dim)
            for j in range(dim)
        )
    )
