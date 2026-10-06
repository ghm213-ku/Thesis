from z3 import *


def get_initial_edges(M_nodes):
    initial_edges = set()
    for i in range(0, M_nodes, 2):
        initial_edges.add((i, i + 1))
    return initial_edges


def build_quantum_maxsat(N_steps, M_nodes, K_nodes, target_edges):
    # opt acts as our MaxSAT / SMT Optimizer
    opt = Optimize()

    initial_edges = get_initial_edges(M_nodes)

    # --- 1. Variable Declarations ---
    # We use z3.Bool() for everything instead of Integer 0/1.
    dummy = [Bool(f"dummy_{n}") for n in range(N_steps)]
    LC = [[Bool(f"LC_{n}_{i}") for i in range(M_nodes)] for n in range(N_steps)]
    CZ = [
        [[Bool(f"CZ_{n}_{i}_{j}") for j in range(M_nodes)] for i in range(M_nodes)]
        for n in range(N_steps)
    ]
    F = [
        [[Bool(f"F_{n}_{i}_{j}") for j in range(M_nodes)] for i in range(M_nodes)]
        for n in range(N_steps)
    ]

    # E-matrix (Adjacency) and L-matrix (Leaders). Only define for j > i to avoid redundancy
    E = [
        [
            [Bool(f"E_{n}_{i}_{j}") if i < j else False for j in range(M_nodes)]
            for i in range(M_nodes)
        ]
        for n in range(N_steps + 1)
    ]
    L = [
        [[Bool(f"L_{n}_{i}_{j}") for j in range(M_nodes)] for i in range(M_nodes)]
        for n in range(N_steps + 1)
    ]

    # Isomorphism variables
    c = [Bool(f"c_{i}") for i in range(M_nodes)]
    p = [[Bool(f"p_{i}_{j}") for j in range(K_nodes)] for i in range(M_nodes)]
    M = [
        [Bool(f"M_{i}_{j}") if i < j else False for j in range(M_nodes)]
        for i in range(M_nodes)
    ]

    # Helper to access symmetric undirected edges cleanly
    def get_E(n, u, v):
        return E[n][min(u, v)][max(u, v)] if u != v else False

    def get_M(u, v):
        return M[min(u, v)][max(u, v)] if u != v else False

    # 1. Initialize the E-matrix with the starting Bell pairs
    for i in range(M_nodes):
        for j in range(i + 1, M_nodes):
            if (i, j) in initial_edges or (j, i) in initial_edges:
                opt.add(E[0][i][j] == True)
            else:
                opt.add(E[0][i][j] == False)

    # 2. Initialize the L-matrix to break symmetry.
    # For each Bell pair (u, v) where u < v, we explicitly make 'u' the leader of the component.
    for x in range(M_nodes):
        for y in range(M_nodes):
            is_leader_match = False
            for u, v in initial_edges:
                leader = min(u, v)
                if x == leader and y in (u, v):
                    is_leader_match = True
                    break
            opt.add(L[0][x][y] == is_leader_match)

    # --- 2. Objective Function ---
    # Multiply floats by 100 to keep it integer optimization (much faster in SAT)
    cost = Sum(
        [
            If(CZ[n][i][j], 317, 0) + If(F[n][i][j], 100, 0) + If(LC[n][i], 1, 0)
            for n in range(N_steps)
            for i in range(M_nodes)
            for j in range(M_nodes)
        ]
    )
    opt.minimize(cost)

    # --- 3. Constraints ---
    for n in range(N_steps):
        # 3.1 Exclusivity & Odd-Even constraint
        active_ops = [dummy[n]] + [LC[n][i] for i in range(M_nodes)]
        for i in range(M_nodes):
            for j in range(M_nodes):
                if i < j:
                    if i % 2 != j % 2:  # Odd-even rule
                        opt.add(Not(CZ[n][i][j]))
                        opt.add(Not(F[n][i][j]))
                    else:
                        active_ops.extend([CZ[n][i][j], F[n][i][j]])
                else:  # Only valid for i < j
                    opt.add(Not(CZ[n][i][j]))
                    opt.add(Not(F[n][i][j]))

        # Exactly one operation active per step
        opt.add(Sum([If(op, 1, 0) for op in active_ops]) == 1)

        # 3.2 Dummy propagation and freeze
        if n < N_steps - 1:
            opt.add(Implies(dummy[n], dummy[n + 1]))

        for i in range(M_nodes):
            for j in range(i + 1, M_nodes):
                opt.add(Implies(dummy[n], E[n + 1][i][j] == E[n][i][j]))

        # 3.3 CZ and F active trackers
        CZ_active = Or(
            [CZ[n][i][j] for i in range(M_nodes) for j in range(i + 1, M_nodes)]
        )
        F_active = Or(
            [F[n][i][j] for i in range(M_nodes) for j in range(i + 1, M_nodes)]
        )

        # 3.4 Leader Logic
        for j in range(M_nodes):
            opt.add(Sum([If(L[n][i][j], 1, 0) for i in range(M_nodes)]) == 1)
            for i in range(M_nodes):
                opt.add(Implies(L[n][i][j], L[n][i][i]))

        for x in range(M_nodes):
            for y in range(M_nodes):
                for z in range(y + 1, M_nodes):
                    # Connected nodes have same leader
                    opt.add(Implies(get_E(n, y, z), L[n][x][y] == L[n][x][z]))

            for i in range(M_nodes):
                for j in range(i + 1, M_nodes):
                    # CZ and F must operate on DIFFERENT leaders
                    opt.add(
                        Implies(
                            Or(CZ[n][i][j], F[n][i][j]),
                            Not(And(L[n][x][i], L[n][x][j])),
                        )
                    )

        # Leader drop count
        opt.add(
            Sum([If(L[n][x][x], 1, 0) for x in range(M_nodes)])
            - Sum([If(L[n + 1][x][x], 1, 0) for x in range(M_nodes)])
            == If(Or(CZ_active, F_active), 1, 0)
        )

        # 3.5 LC Operation (Native XOR Edge Flips)
        for i in range(M_nodes):
            for j1 in range(M_nodes):
                for j2 in range(j1 + 1, M_nodes):
                    if i != j1 and i != j2:
                        flip_cond = Xor(
                            get_E(n, j1, j2), And(get_E(n, i, j1), get_E(n, i, j2))
                        )
                        opt.add(Implies(LC[n][i], get_E(n + 1, j1, j2) == flip_cond))
                    else:
                        # Direct edges to LC operated node remain frozen
                        opt.add(
                            Implies(LC[n][i], get_E(n + 1, j1, j2) == get_E(n, j1, j2))
                        )

        # 3.6 CZ Edge Updates
        for i in range(M_nodes):
            for j in range(i + 1, M_nodes):
                opt.add(Implies(CZ[n][i][j], get_E(n + 1, i, j) == True))
                for a in range(M_nodes):
                    for b in range(a + 1, M_nodes):
                        if {a, b} != {i, j}:
                            opt.add(
                                Implies(
                                    CZ[n][i][j], get_E(n + 1, a, b) == get_E(n, a, b)
                                )
                            )

        # 3.7 Fusion Edge Updates
        for i in range(M_nodes):
            for j in range(i + 1, M_nodes):
                opt.add(Implies(F[n][i][j], get_E(n + 1, i, j) == True))
                for k in range(M_nodes):
                    if k != i and k != j:
                        # Neighborhood of j is transferred/XORed to i
                        opt.add(
                            Implies(
                                F[n][i][j],
                                get_E(n + 1, i, k)
                                == Or(get_E(n, i, k), get_E(n, j, k)),
                            )
                        )
                        # j is disconnected
                        opt.add(Implies(F[n][i][j], get_E(n + 1, j, k) == False))
                # Edges disjoint from {i,j} freeze
                for a in range(M_nodes):
                    for b in range(a + 1, M_nodes):
                        if a not in {i, j} and b not in {i, j}:
                            opt.add(
                                Implies(
                                    F[n][i][j], get_E(n + 1, a, b) == get_E(n, a, b)
                                )
                            )

    # --- 4. Vertex Deletion & Isomorphism ---
    for i in range(M_nodes):
        for j in range(i + 1, M_nodes):
            # Native AND for vertex deletion
            opt.add(M[i][j] == And(get_E(N_steps, i, j), c[i], c[j]))

    for j in range(K_nodes):  # Every target logical node must be mapped to exactly once
        opt.add(Sum([If(p[i][j], 1, 0) for i in range(M_nodes)]) == 1)

    for i in range(M_nodes):
        # A chosen node maps to exactly one logical node, unchosen to zero
        opt.add(Sum([If(p[i][j], 1, 0) for j in range(K_nodes)]) == If(c[i], 1, 0))

    # Evaluate Isomorphism dynamically
    for i in range(M_nodes):
        for j in range(K_nodes):
            # Sum of physical edges mapping into this node
            physical_edges = Sum(
                [If(And(get_M(i, k), p[k][j]), 1, 0) for k in range(M_nodes) if k != i]
            )
            # Sum of logical target edges
            logical_edges = Sum(
                [
                    If(p[i][l], 1, 0)
                    for l in range(K_nodes)
                    if l != j and ((min(l, j), max(l, j)) in target_edges)
                ]
            )
            opt.add(physical_edges == logical_edges)

    return opt, dummy, LC, CZ, F, c, p


# ----- Example Usage -----
if __name__ == "__main__":
    # Example Target Graph: A simple 3-node path (0-1-2)
    target_edges = {
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
    }  # Adjust as needed

    # Configure variables (scale these up to test limit)
    N_steps = 6
    M_nodes = 10
    K_nodes = 10

    print("Building model...")
    opt, dummy, LC, CZ, F, c, p = build_quantum_maxsat(
        N_steps, M_nodes, K_nodes, target_edges
    )

    print("Solving...")
    result = opt.check()

    if result == sat:
        print("\nOptimal sequence found!\n")
        m = opt.model()  # Extract the satisfied model

        print("--- Sequence of Operations ---")
        for n in range(N_steps):
            if is_true(m.evaluate(dummy[n])):
                print(f"Step {n}: Dummy (Wait)")
                continue

            for i in range(M_nodes):
                if is_true(m.evaluate(LC[n][i])):
                    print(f"Step {n}: LC on node {i}")
                    break
                for j in range(i + 1, M_nodes):
                    if is_true(m.evaluate(CZ[n][i][j])):
                        print(f"Step {n}: CZ between {i} and {j}")
                        break
                    if is_true(m.evaluate(F[n][i][j])):
                        print(f"Step {n}: Fusion between {i} and {j}")
                        break

        print("\n--- Final Node Mapping (Physical -> Target) ---")
        for i in range(M_nodes):
            if is_true(m.evaluate(c[i])):
                for j in range(K_nodes):
                    if is_true(m.evaluate(p[i][j])):
                        print(f"Physical Node {i} -> Target Node {j}")
            else:
                print(f"Physical Node {i} -> [Deleted via Postselection]")
    else:
        print("\nNo feasible sequence exists within given steps.")
