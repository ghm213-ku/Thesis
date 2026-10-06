from ortools.sat.python import cp_model
from collections import defaultdict
import time

def build_quantum_cpsat(
    N_steps: int,
    M_nodes: int,
    K_nodes: int,
    initial_edges: set,
    target_edges: set,
    parity_rule: str = "same"  # "same" (i % 2 == j % 2) or "different" (i % 2 != j % 2)
):
    model = cp_model.CpModel()

    # --- 1. Variable Declarations ---
    dummy = [model.NewBoolVar(f"dummy_{n}") for n in range(N_steps)]
    LC = [[model.NewBoolVar(f"LC_{n}_{i}") for i in range(M_nodes)] for n in range(N_steps)]
    
    CZ = {}
    for n in range(N_steps):
        for i in range(M_nodes):
            for j in range(i + 1, M_nodes):
                CZ[n, i, j] = model.NewBoolVar(f"CZ_{n}_{i}_{j}")
                
    F = {}
    for n in range(N_steps):
        for i in range(M_nodes):
            for j in range(M_nodes):
                if i != j:
                    F[n, i, j] = model.NewBoolVar(f"F_{n}_{i}_{j}")

    # E-matrix: undirected edges, defined only for i < j
    E = {}
    for n in range(N_steps + 1):
        for i in range(M_nodes):
            for j in range(i + 1, M_nodes):
                E[n, i, j] = model.NewBoolVar(f"E_{n}_{i}_{j}")

    def get_E(n, u, v):
        if u == v:
            return None
        return E[n, min(u, v), max(u, v)]

    # L-matrix: leaders
    L = {}
    for n in range(N_steps + 1):
        for i in range(M_nodes):
            for j in range(M_nodes):
                L[n, i, j] = model.NewBoolVar(f"L_{n}_{i}_{j}")

    # Isomorphism variables
    c = [model.NewBoolVar(f"c_{i}") for i in range(M_nodes)]
    p = [[model.NewBoolVar(f"p_{i}_{j}") for j in range(K_nodes)] for i in range(M_nodes)]
    M_mat = {}
    for i in range(M_nodes):
        for j in range(i + 1, M_nodes):
            M_mat[i, j] = model.NewBoolVar(f"M_{i}_{j}")

    def get_M(u, v):
        if u == v:
            return None
        return M_mat[min(u, v), max(u, v)]

    # --- 2. Initial State Boundary Conditions (n = 0) ---
    for i in range(M_nodes):
        for j in range(i + 1, M_nodes):
            if (i, j) in initial_edges or (j, i) in initial_edges:
                model.Add(E[0, i, j] == 1)
            else:
                model.Add(E[0, i, j] == 0)

    # Pin initial leaders to min(u, v) for each Bell pair to break root symmetry
    for x in range(M_nodes):
        for y in range(M_nodes):
            is_match = False
            for (u, v) in initial_edges:
                leader = min(u, v)
                if x == leader and y in (u, v):
                    is_match = True
                    break
            model.Add(L[0, x, y] == (1 if is_match else 0))

    # --- 3. Step Constraints (0 to N-1) ---
    for n in range(N_steps):
        # 3.1 Parity restriction & Operation Exclusivity
        valid_ops = [dummy[n]] + [LC[n][i] for i in range(M_nodes)]

        for i in range(M_nodes):
            for j in range(i + 1, M_nodes):
                is_valid_parity = (i % 2 == j % 2) if parity_rule == "same" else (i % 2 != j % 2)
                if is_valid_parity:
                    valid_ops.append(CZ[n, i, j])
                else:
                    model.Add(CZ[n, i, j] == 0)

        for i in range(M_nodes):
            for j in range(M_nodes):
                if i != j:
                    is_valid_parity = (i % 2 == j % 2) if parity_rule == "same" else (i % 2 != j % 2)
                    if is_valid_parity:
                        valid_ops.append(F[n, i, j])
                    else:
                        model.Add(F[n, i, j] == 0)

        model.AddExactlyOne(valid_ops)

        # 3.2 Dummy Propagation and Edge Freeze
        if n < N_steps - 1:
            model.AddImplication(dummy[n], dummy[n + 1])

        for i in range(M_nodes):
            for j in range(i + 1, M_nodes):
                model.Add(E[n + 1, i, j] == E[n, i, j]).OnlyEnforceIf(dummy[n])
            for j in range(M_nodes):
                model.Add(L[n + 1, i, j] == L[n, i, j]).OnlyEnforceIf(dummy[n])

        # 3.3 Leader Consistency at step n
        for j in range(M_nodes):
            model.AddExactlyOne([L[n, i, j] for i in range(M_nodes)])
            for i in range(M_nodes):
                model.AddImplication(L[n, i, j], L[n, i, i])

        for y in range(M_nodes):
            for z in range(y + 1, M_nodes):
                for x in range(M_nodes):
                    model.Add(L[n, x, y] == L[n, x, z]).OnlyEnforceIf(E[n, y, z])

        # CZ and F require operands from different components
        for i in range(M_nodes):
            for j in range(i + 1, M_nodes):
                if (n, i, j) in CZ:
                    for x in range(M_nodes):
                        model.Add(L[n, x, i] + L[n, x, j] <= 1).OnlyEnforceIf(CZ[n, i, j])

        for i in range(M_nodes):
            for j in range(M_nodes):
                if (n, i, j) in F:
                    for x in range(M_nodes):
                        model.Add(L[n, x, i] + L[n, x, j] <= 1).OnlyEnforceIf(F[n, i, j])

        # Leader count drops by 1 on CZ or F
        active_gate = model.NewBoolVar(f"active_gate_{n}")
        gate_terms = [CZ[n, i, j] for i in range(M_nodes) for j in range(i + 1, M_nodes) if (n, i, j) in CZ] + \
                     [F[n, i, j] for i in range(M_nodes) for j in range(M_nodes) if (n, i, j) in F]
        model.Add(active_gate == sum(gate_terms))
        model.Add(sum(L[n, x, x] for x in range(M_nodes)) - sum(L[n + 1, x, x] for x in range(M_nodes)) == active_gate)

        # 3.4 Local Complementation (LC)
        for i in range(M_nodes):
            for k in range(M_nodes):
                if k != i:
                    model.Add(get_E(n + 1, i, k) == get_E(n, i, k)).OnlyEnforceIf(LC[n][i])

            for j1 in range(M_nodes):
                for j2 in range(j1 + 1, M_nodes):
                    if i != j1 and i != j2:
                        prod = model.NewBoolVar(f"lc_prod_{n}_{i}_{j1}_{j2}")
                        model.AddBoolAnd([get_E(n, i, j1), get_E(n, i, j2)]).OnlyEnforceIf(prod)
                        model.AddBoolOr([get_E(n, i, j1).Not(), get_E(n, i, j2).Not()]).OnlyEnforceIf(prod.Not())

                        model.Add(get_E(n + 1, j1, j2) != get_E(n, j1, j2)).OnlyEnforceIf([LC[n][i], prod])
                        model.Add(get_E(n + 1, j1, j2) == get_E(n, j1, j2)).OnlyEnforceIf([LC[n][i], prod.Not()])

            for x in range(M_nodes):
                for y in range(M_nodes):
                    model.Add(L[n + 1, x, y] == L[n, x, y]).OnlyEnforceIf(LC[n][i])

        # 3.5 CZ Operations
        for i in range(M_nodes):
            for j in range(i + 1, M_nodes):
                if (n, i, j) in CZ:
                    model.Add(get_E(n + 1, i, j) == 1).OnlyEnforceIf(CZ[n, i, j])
                    for a in range(M_nodes):
                        for b in range(a + 1, M_nodes):
                            if (a, b) != (i, j):
                                model.Add(get_E(n + 1, a, b) == get_E(n, a, b)).OnlyEnforceIf(CZ[n, i, j])

        # 3.6 Fusion Operations
        for i in range(M_nodes):
            for j in range(M_nodes):
                if (n, i, j) in F:
                    model.Add(get_E(n + 1, i, j) == 1).OnlyEnforceIf(F[n, i, j])

                    for k in range(M_nodes):
                        if k != i and k != j:
                            model.Add(get_E(n + 1, j, k) == 0).OnlyEnforceIf(F[n, i, j])
                            model.AddBoolOr([get_E(n, i, k), get_E(n, j, k)]).OnlyEnforceIf([F[n, i, j], get_E(n + 1, i, k)])
                            model.Add(get_E(n + 1, i, k) == 1).OnlyEnforceIf([F[n, i, j], get_E(n, i, k)])
                            model.Add(get_E(n + 1, i, k) == 1).OnlyEnforceIf([F[n, i, j], get_E(n, j, k)])
                            model.Add(get_E(n + 1, i, k) == 0).OnlyEnforceIf([F[n, i, j], get_E(n, i, k).Not(), get_E(n, j, k).Not()])

                    for a in range(M_nodes):
                        for b in range(a + 1, M_nodes):
                            if a not in (i, j) and b not in (i, j):
                                model.Add(get_E(n + 1, a, b) == get_E(n, a, b)).OnlyEnforceIf(F[n, i, j])

        # 3.7 Symmetry Breaking Cuts
        if n < N_steps - 1:
            for i in range(M_nodes):
                model.Add(LC[n][i] + LC[n + 1][i] <= 1)
            for i in range(M_nodes):
                for j in range(i + 1, M_nodes):
                    if (n, i, j) in CZ and (n + 1, i, j) in CZ:
                        model.Add(CZ[n, i, j] + CZ[n + 1, i, j] <= 1)

    # --- 4. Subgraph Isomorphism & Vertex Deletion ---
    for i in range(M_nodes):
        for j in range(i + 1, M_nodes):
            model.AddBoolAnd([E[N_steps, i, j], c[i], c[j]]).OnlyEnforceIf(M_mat[i, j])
            model.AddBoolOr([E[N_steps, i, j].Not(), c[i].Not(), c[j].Not()]).OnlyEnforceIf(M_mat[i, j].Not())

    # Target cardinality preservation
    model.Add(sum(c) == K_nodes)
    model.Add(sum(M_mat[i, j] for i in range(M_nodes) for j in range(i + 1, M_nodes)) == len(target_edges))

    for j in range(K_nodes):
        model.AddExactlyOne([p[i][j] for i in range(M_nodes)])

    for i in range(M_nodes):
        model.Add(sum(p[i][j] for j in range(K_nodes)) == c[i])

    # Degree preservation cuts
    target_degrees = defaultdict(int)
    for u, v in target_edges:
        target_degrees[u] += 1
        target_degrees[v] += 1

    for i in range(M_nodes):
        deg_i = sum(get_M(i, k) for k in range(M_nodes) if k != i)
        target_deg_sum = sum(target_degrees[j] * p[i][j] for j in range(K_nodes))
        model.Add(deg_i == target_deg_sum)

    # Structural adjacency mapping
    for i in range(M_nodes):
        for k in range(M_nodes):
            if i != k:
                for j in range(K_nodes):
                    Z = model.NewBoolVar(f"Z_{i}_{k}_{j}")
                    model.AddBoolAnd([get_M(i, k), p[k][j]]).OnlyEnforceIf(Z)
                    model.AddBoolOr([get_M(i, k).Not(), p[k][j].Not()]).OnlyEnforceIf(Z.Not())

    for i in range(M_nodes):
        for j in range(K_nodes):
            lhs = sum(model.NewBoolVar(f"Z_tmp_{i}_{k}_{j}") for k in range(M_nodes) if k != i)
            # Recompute direct Z terms for cleaner expression
            z_terms = []
            for k in range(M_nodes):
                if k != i:
                    z_var = model.NewBoolVar(f"Z_iso_{i}_{k}_{j}")
                    model.AddBoolAnd([get_M(i, k), p[k][j]]).OnlyEnforceIf(z_var)
                    model.AddBoolOr([get_M(i, k).Not(), p[k][j].Not()]).OnlyEnforceIf(z_var.Not())
                    z_terms.append(z_var)

            rhs = sum(
                p[i][l] for l in range(K_nodes)
                if l != j and ((min(l, j), max(l, j)) in target_edges or (max(l, j), min(l, j)) in target_edges)
            )
            model.Add(sum(z_terms) == rhs)

    # --- 5. Objective Function ---
    cost_terms = []
    for n in range(N_steps):
        for i in range(M_nodes):
            cost_terms.append(1 * LC[n][i])
            for j in range(i + 1, M_nodes):
                if (n, i, j) in CZ:
                    cost_terms.append(317 * CZ[n, i, j])
            for j in range(M_nodes):
                if (n, i, j) in F:
                    cost_terms.append(100 * F[n, i, j])

    model.Minimize(sum(cost_terms))

    return model, dummy, LC, CZ, F, c, p


# --- Execution and Output ---
if __name__ == "__main__":
    # Example Setup
    target_edges = {(0, 1), (1, 2)}             # 3-node path
    initial_edges = {(0, 1), (2, 3), (4, 5)}    # 3 Bell pairs (6 physical nodes)
    
    N_steps = 5
    M_nodes = 6
    K_nodes = 3

    print(f"Building CP-SAT model (Nodes: {M_nodes}, Steps: {N_steps}, Target size: {K_nodes})...")
    start_build = time.time()
    model, dummy, LC, CZ, F, c, p = build_quantum_cpsat(
        N_steps, M_nodes, K_nodes, initial_edges, target_edges, parity_rule="same"
    )
    print(f"Model built in {time.time() - start_build:.2f}s")

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 8       # Utilize multi-core parallelism
    solver.parameters.max_time_in_seconds = 300.0   # 5-minute timeout
    solver.parameters.log_search_progress = True    # Real-time search log

    print("Starting CP-SAT solve...\n")
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        status_name = "Optimal" if status == cp_model.OPTIMAL else "Feasible"
        print(f"\n{status_name} solution found!")
        print(f"Objective Cost: {solver.ObjectiveValue() / 100.0:.2f}")
        print(f"Wall Time: {solver.WallTime():.2f}s\n")

        print("--- Sequence of Operations ---")
        for n in range(N_steps):
            if solver.BooleanValue(dummy[n]):
                print(f"Step {n}: [Dummy / Wait]")
                continue

            for i in range(M_nodes):
                if solver.BooleanValue(LC[n][i]):
                    print(f"Step {n}: LC on physical node {i}")
                    break

            for i in range(M_nodes):
                for j in range(i + 1, M_nodes):
                    if (n, i, j) in CZ and solver.BooleanValue(CZ[n, i, j]):
                        print(f"Step {n}: CZ between {i} and {j}")
                        break

            for i in range(M_nodes):
                for j in range(M_nodes):
                    if (n, i, j) in F and solver.BooleanValue(F[n, i, j]):
                        print(f"Step {n}: Fusion({i} <- {j}) [transferring {j} into {i}]")
                        break

        print("\n--- Final Subgraph Mapping ---")
        for i in range(M_nodes):
            if solver.BooleanValue(c[i]):
                for j in range(K_nodes):
                    if solver.BooleanValue(p[i][j]):
                        print(f"Physical Node {i} -> Logical Target Node {j}")
            else:
                print(f"Physical Node {i} -> [Deleted via Postselection]")
    else:
        print("\nNo feasible sequence found within given steps or timeout.")