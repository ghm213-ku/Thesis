import pulp
import gurobipy as gp
from gurobipy import GRB


class ILPModel:
    def __init__(self, N, M, E_0, L_0, T):
        """
        N: Number of timeline operations (max index). Timeline goes from 0 to N.
        M: Max index of physical qubits. Qubits are 0 to M.
        E_0: Initial adjacency matrix (M+1 x M+1).
        L_0: Initial DSU leader matrix (M+1 x M+1).
        T: Logical target adjacency matrix (K x K).
        """
        self.N = N
        self.M = M
        self.K_nodes = len(T)
        self.E_0 = E_0
        self.L_0 = L_0
        self.T = T

        # The final state of the graph after N operations is at N+1
        self.n_target = self.N + 1

        self.prob = pulp.LpProblem("Quantum_Circuit_Routing", pulp.LpMinimize)

    def build(self):
        """Master build method. Acts as the table of contents for the model."""
        self._define_variables()
        self._define_objective()

        self._add_boundary_conditions()
        self._add_global_operations()
        self._add_lc_constraints()
        self._add_cz_and_f_shared_logic()
        self._add_f_edge_logic()
        self._add_isomorphism_and_deletion()

        return self.prob

    def _define_variables(self):
        # 1. Timeline Operation Variables
        self.y_dummy = pulp.LpVariable.dicts("y_dummy", range(self.N + 1), cat="Binary")
        self.y_LC = pulp.LpVariable.dicts(
            "y_LC", (range(self.N + 1), range(self.M + 1)), cat="Binary"
        )
        self.y_CZ = pulp.LpVariable.dicts(
            "y_CZ",
            (range(self.N + 1), range(self.M + 1), range(self.M + 1)),
            cat="Binary",
        )
        self.y_F = pulp.LpVariable.dicts(
            "y_F",
            (range(self.N + 1), range(self.M + 1), range(self.M + 1)),
            cat="Binary",
        )

        self.y_CZ_active = pulp.LpVariable.dicts(
            "y_CZ_active", range(self.N + 1), cat="Binary"
        )
        self.y_F_active = pulp.LpVariable.dicts(
            "y_F_active", range(self.N + 1), cat="Binary"
        )

        # 2. State Matrices
        self.E = pulp.LpVariable.dicts(
            "E", (range(self.N + 2), range(self.M + 1), range(self.M + 1)), cat="Binary"
        )
        self.L = pulp.LpVariable.dicts(
            "L", (range(self.N + 2), range(self.M + 1), range(self.M + 1)), cat="Binary"
        )

        # 3. Vertex Deletion & Isomorphism Variables
        self.c = pulp.LpVariable.dicts("c", range(self.M + 1), cat="Binary")
        self.Mask = pulp.LpVariable.dicts(
            "Mask", (range(self.M + 1), range(self.M + 1)), cat="Binary"
        )
        self.p = pulp.LpVariable.dicts(
            "p", (range(self.M + 1), range(self.K_nodes)), cat="Binary"
        )

        # Continuous Z bounds for McCormick envelope
        self.Z = pulp.LpVariable.dicts(
            "Z",
            (range(self.M + 1), range(self.M + 1), range(self.K_nodes)),
            lowBound=0,
            cat="Continuous",
        )

        # Continuous auxiliary variables for LC McCormick envelopes
        # Indexed by [n][i][j1][j2]
        self.l_aux = pulp.LpVariable.dicts(
            "l_aux",
            (
                range(self.N + 1),
                range(self.M + 1),
                range(self.M + 1),
                range(self.M + 1),
            ),
            lowBound=0,
            cat="Continuous",
        )

        self.z_aux = pulp.LpVariable.dicts(
            "z_aux",
            (
                range(self.N + 1),
                range(self.M + 1),
                range(self.M + 1),
                range(self.M + 1),
            ),
            lowBound=0,
            cat="Continuous",
        )

        self.f_aux = pulp.LpVariable.dicts(
            "f_aux",
            (
                range(self.N + 1),
                range(self.M + 1),
                range(self.M + 1),
                range(self.M + 1),
            ),
            lowBound=0,
            cat="Continuous",
        )

    def _define_objective(self):
        self.prob += (
            pulp.lpSum(
                [
                    3.17 * self.y_CZ[n][i][j] + 1.0 * self.y_F[n][i][j]
                    for n in range(self.N + 1)
                    for i in range(self.M + 1)
                    for j in range(self.M + 1)
                ]
            )
            + pulp.lpSum(
                [
                    0.01 * self.y_LC[n][i]
                    for n in range(self.N + 1)
                    for i in range(self.M + 1)
                ]
            ),
            "Minimize_Cost",
        )

    def _add_boundary_conditions(self):
        for i in range(self.M + 1):
            for j in range(self.M + 1):
                self.prob += self.E[0][i][j] == self.E_0[i][j], f"Init_E_{i}_{j}"
                self.prob += self.L[0][i][j] == self.L_0[i][j], f"Init_L_{i}_{j}"

        # IMPORTANT: E[N+1] == target is explicitly OMITTED here to let the
        # isomorphism block handle the dynamic mapping dynamically.

    def _add_global_operations(self):
        for n in range(self.N + 1):

            # One operation per step
            self.prob += (
                self.y_dummy[n]
                + pulp.lpSum([self.y_LC[n][i] for i in range(self.M + 1)])
                + pulp.lpSum(
                    [
                        self.y_CZ[n][i][j] + self.y_F[n][i][j]
                        for i in range(self.M + 1)
                        for j in range(self.M + 1)
                    ]
                )
                == 1,
                f"One_Op_{n}",
            )

            # Dummy monotonicity
            if n < self.N:
                self.prob += self.y_dummy[n + 1] >= self.y_dummy[n], f"Dummy_Mono_{n}"

            # Dummy E-Matrix inertia
            for i in range(self.M + 1):
                for j in range(i + 1, self.M + 1):
                    self.prob += (
                        self.E[n + 1][i][j] - self.E[n][i][j] <= 1 - self.y_dummy[n],
                        f"Dum_E_UB_{n}_{i}_{j}",
                    )
                    self.prob += (
                        self.E[n][i][j] - self.E[n + 1][i][j] <= 1 - self.y_dummy[n],
                        f"Dum_E_LB_{n}_{i}_{j}",
                    )

            # Parity checks
            for i in range(self.M + 1):
                for j in range(self.M + 1):
                    if i % 2 != j % 2:
                        self.prob += self.y_CZ[n][i][j] == 0, f"Parity_CZ_{n}_{i}_{j}"
                        self.prob += self.y_F[n][i][j] == 0, f"Parity_F_{n}_{i}_{j}"

        # Global Structural Matrices
        for n in range(self.N + 2):
            for j in range(self.M + 1):
                self.prob += (
                    pulp.lpSum([self.L[n][i][j] for i in range(self.M + 1)]) == 1,
                    f"DSU_Col_{n}_{j}",
                )

            for i in range(self.M + 1):
                for j in range(self.M + 1):
                    self.prob += (
                        self.L[n][i][j] <= self.L[n][i][i],
                        f"DSU_Diag_{n}_{i}_{j}",
                    )

                # Enforce strictly upper triangular E-matrix
                for j in range(i + 1):
                    self.prob += self.E[n][i][j] == 0, f"E_Lower_{n}_{i}_{j}"

        for n in range(self.N + 1):
            for i in range(self.M + 1):
                for j in range(self.M + 1):
                    for x in range(self.M + 1):
                        self.prob += (
                            self.y_CZ[n][i][j] + self.y_F[n][i][j]
                            <= 2 - self.L[n][x][j] - self.L[n][x][i],
                            f"CZ_F_Deny_{n}_{i}_{j}_{x}",
                        )

        for n in range(self.N + 2):
            for x in range(self.M + 1):
                for y in range(self.M + 1):
                    for z in range(
                        y + 1, self.M + 1
                    ):  # strictly y < z to match upper-triangular E
                        self.prob += (
                            self.L[n][x][y] - self.L[n][x][z] <= 1 - self.E[n][y][z],
                            f"Edge_L_UB_{n}_{x}_{y}_{z}",
                        )
                        self.prob += (
                            self.L[n][x][z] - self.L[n][x][y] <= 1 - self.E[n][y][z],
                            f"Edge_L_LB_{n}_{x}_{y}_{z}",
                        )

    def _add_lc_constraints(self):
        for n in range(self.N + 1):
            for i in range(self.M + 1):

                # 1. Freeze edges directly connected to LC target
                for j in range(i + 1, self.M + 1):
                    self.prob += (
                        self.E[n + 1][i][j] - self.E[n][i][j]
                        <= 1 - self.y_LC[n][i] - self.y_LC[n][j],
                        f"LC_Frz_UB_{n}_{i}_{j}",
                    )
                    self.prob += (
                        self.E[n][i][j] - self.E[n + 1][i][j]
                        <= 1 - self.y_LC[n][i] - self.y_LC[n][j],
                        f"LC_Frz_LB_{n}_{i}_{j}",
                    )

                # 2. LC Edge Flip Logic via McCormick Envelopes
                for j1 in range(self.M + 1):
                    for j2 in range(j1 + 1, self.M + 1):  # Strictly j1 < j2
                        if i != j1 and i != j2:

                            e1 = self.E[n][min(i, j1)][max(i, j1)]
                            e2 = self.E[n][min(i, j2)][max(i, j2)]
                            yLC = self.y_LC[n][i]

                            l_var = self.l_aux[n][i][j1][j2]
                            z_var = self.z_aux[n][i][j1][j2]

                            # l = E_n(i, j1) AND E_n(i, j2)
                            self.prob += l_var <= e1, f"LC_l_UB1_{n}_{i}_{j1}_{j2}"
                            self.prob += l_var <= e2, f"LC_l_UB2_{n}_{i}_{j1}_{j2}"
                            self.prob += (
                                l_var >= e2 + e1 - 1,
                                f"LC_l_LB_{n}_{i}_{j1}_{j2}",
                            )

                            # z = l AND y_LC(i)
                            self.prob += z_var <= l_var, f"LC_z_UB1_{n}_{i}_{j1}_{j2}"
                            self.prob += z_var <= yLC, f"LC_z_UB2_{n}_{i}_{j1}_{j2}"
                            self.prob += (
                                z_var >= l_var + yLC - 1,
                                f"LC_z_LB_{n}_{i}_{j1}_{j2}",
                            )

                            e_n1_j = self.E[n + 1][j1][j2]
                            e_n_j = self.E[n][j1][j2]

                            # Edge Flip Bounds
                            self.prob += (
                                e_n1_j + e_n_j >= z_var,
                                f"LC_Flip_LB1_{n}_{i}_{j1}_{j2}",
                            )
                            self.prob += (
                                e_n1_j + e_n_j <= 2 - z_var,
                                f"LC_Flip_UB1_{n}_{i}_{j1}_{j2}",
                            )
                            self.prob += (
                                e_n1_j - e_n_j <= z_var + 1 - yLC,
                                f"LC_Flip_UB2_{n}_{i}_{j1}_{j2}",
                            )
                            self.prob += (
                                e_n_j - e_n1_j <= z_var + 1 - yLC,
                                f"LC_Flip_LB2_{n}_{i}_{j1}_{j2}",
                            )

    def _add_cz_and_f_shared_logic(self):
        for n in range(self.N + 1):

            # Active Flags
            self.prob += (
                self.y_CZ_active[n]
                == pulp.lpSum(
                    [
                        self.y_CZ[n][a][b]
                        for a in range(self.M + 1)
                        for b in range(self.M + 1)
                    ]
                ),
                f"CZ_Active_{n}",
            )
            self.prob += (
                self.y_F_active[n]
                == pulp.lpSum(
                    [
                        self.y_F[n][a][b]
                        for a in range(self.M + 1)
                        for b in range(self.M + 1)
                    ]
                ),
                f"F_Active_{n}",
            )

            # Conservation of Leaders (One death per gate)
            self.prob += (
                pulp.lpSum(
                    [self.L[n][x][x] - self.L[n + 1][x][x] for x in range(self.M + 1)]
                )
                == self.y_CZ_active[n] + self.y_F_active[n],
                f"Lead_Kill_{n}",
            )

            # Base CZ & F Edge Formation between active nodes (i, j)
            for i in range(self.M + 1):
                for j in range(self.M + 1):
                    if i != j:
                        self.prob += (
                            self.E[n + 1][min(i, j)][max(i, j)]
                            >= self.y_CZ[n][i][j] + self.y_F[n][i][j],
                            f"CZ_F_Base_{n}_{i}_{j}",
                        )

            # Localized CZ Inertia
            for j1 in range(self.M + 1):
                for j2 in range(j1 + 1, self.M + 1):
                    rhs_cz = (
                        self.y_CZ[n][j1][j2]
                        + self.y_CZ[n][j2][j1]
                        + 1
                        - self.y_CZ_active[n]
                    )
                    self.prob += (
                        self.E[n + 1][j1][j2] - self.E[n][j1][j2] <= rhs_cz,
                        f"CZ_Inert_UB_{n}_{j1}_{j2}",
                    )
                    self.prob += (
                        self.E[n][j1][j2] - self.E[n + 1][j1][j2] <= rhs_cz,
                        f"CZ_Inert_LB_{n}_{j1}_{j2}",
                    )

            for a in range(self.M + 1):
                for b in range(a + 1, self.M + 1):
                    f_sum = pulp.lpSum(
                        [
                            self.y_F[n][a][x]
                            + self.y_F[n][x][a]
                            + self.y_F[n][b][x]
                            + self.y_F[n][x][b]
                            for x in range(self.M + 1)
                        ]
                    )
                    rhs_inertia = (
                        self.y_CZ[n][a][b]
                        + f_sum
                        + 1
                        - self.y_CZ_active[n]
                        - self.y_F_active[n]
                    )

                    self.prob += (
                        self.E[n + 1][a][b] - self.E[n][a][b] <= rhs_inertia,
                        f"Glb_E_UB_{n}_{a}_{b}",
                    )
                    self.prob += (
                        self.E[n][a][b] - self.E[n + 1][a][b] <= rhs_inertia,
                        f"Glb_E_LB_{n}_{a}_{b}",
                    )

    def _add_f_edge_logic(self):
        for n in range(self.N + 1):

            # 1. McCormick Envelopes for f_aux
            for i in range(self.M + 1):
                for j in range(self.M + 1):
                    if i != j:
                        for k in range(self.M + 1):
                            if k != i and k != j:
                                f_var = self.f_aux[n][i][j][k]
                                yF = self.y_F[n][i][j]
                                e_jk = self.E[n][min(j, k)][max(j, k)]

                                self.prob += f_var <= yF, f"f_UB_y_{n}_{i}_{j}_{k}"
                                self.prob += f_var <= e_jk, f"f_UB_E_{n}_{i}_{j}_{k}"
                                self.prob += (
                                    f_var >= e_jk - 1 + yF,
                                    f"f_LB_{n}_{i}_{j}_{k}",
                                )

            # 2. Localized F-Inertia & Spawning/Destruction
            for a in range(self.M + 1):
                for b in range(a + 1, self.M + 1):

                    # Spawning sum: (a,b) acts as physical edge (i,k). We sum over all possible j.
                    sum_f_spawn = pulp.lpSum(
                        [
                            self.f_aux[n][a][j][b]
                            for j in range(self.M + 1)
                            if j != a and j != b
                        ]
                        + [
                            self.f_aux[n][b][j][a]
                            for j in range(self.M + 1)
                            if j != a and j != b
                        ]
                    )

                    # Destruction sum: (a,b) acts as physical edge (j,k). We sum over all possible i.
                    sum_f_destroy = pulp.lpSum(
                        [
                            self.f_aux[n][i][a][b]
                            for i in range(self.M + 1)
                            if i != a and i != b
                        ]
                        + [
                            self.f_aux[n][i][b][a]
                            for i in range(self.M + 1)
                            if i != a and i != b
                        ]
                    )

                    e_n1 = self.E[n + 1][a][b]
                    e_n = self.E[n][a][b]
                    yF_act = self.y_F_active[n]

                    # Spawning rules for edge (i,k)
                    self.prob += (
                        e_n1 - e_n <= sum_f_spawn + 1 - yF_act,
                        f"F_Spawn_UB_{n}_{a}_{b}",
                    )
                    self.prob += (
                        e_n1 - e_n >= sum_f_spawn - 1 + yF_act,
                        f"F_Spawn_LB_{n}_{a}_{b}",
                    )

                    # Destruction rules for edge (j,k)
                    self.prob += (
                        e_n - e_n1 <= sum_f_destroy + 1 - yF_act,
                        f"F_Destroy_UB_{n}_{a}_{b}",
                    )
                    self.prob += (
                        e_n - e_n1 >= sum_f_destroy - 1 + yF_act,
                        f"F_Destroy_LB_{n}_{a}_{b}",
                    )

    def _add_isomorphism_and_deletion(self):

        # 1. Precompute target graph properties (Strictly Upper Triangular)
        target_degrees = []
        for j in range(self.K_nodes):
            # Sum column j (edges to smaller indices) + row j (edges to larger indices)
            deg = sum([self.T[l][j] for l in range(j)]) + sum(
                [self.T[j][l] for l in range(j + 1, self.K_nodes)]
            )
            target_degrees.append(deg)

        # In an upper triangular matrix, every edge is represented exactly once
        total_target_edges = sum(
            [
                self.T[a][b]
                for a in range(self.K_nodes)
                for b in range(a + 1, self.K_nodes)
            ]
        )

        # 3. Global Edge Count Match (O(1) prune)
        self.prob += (
            pulp.lpSum(
                [
                    self.Mask[i][j]
                    for i in range(self.M + 1)
                    for j in range(i + 1, self.M + 1)
                ]
            )
            == total_target_edges,
            "Global_Edge_Count",
        )

        # 5. Exact Degree Matching (Aggressive Branch Pruning)
        for i in range(self.M + 1):
            # Physical degree calculation (Upper-triangular aware)
            physical_deg = pulp.lpSum([self.Mask[k][i] for k in range(i)]) + pulp.lpSum(
                [self.Mask[i][k] for k in range(i + 1, self.M + 1)]
            )

            # Map the physical degree to the exact assigned target degree
            assigned_target_deg = pulp.lpSum(
                [self.p[i][j] * target_degrees[j] for j in range(self.K_nodes)]
            )

            self.prob += physical_deg == assigned_target_deg, f"Degree_Match_{i}"

        for i in range(self.M + 1):
            for j in range(i + 1, self.M + 1):
                # Mask bounded by final physical E-matrix state
                self.prob += (
                    self.Mask[i][j] <= self.E[self.n_target][i][j],
                    f"Mask_UB_E_{i}_{j}",
                )
                self.prob += self.Mask[i][j] <= self.c[i], f"Mask_UB_c_i_{i}_{j}"
                self.prob += self.Mask[i][j] <= self.c[j], f"Mask_UB_c_j_{i}_{j}"
                self.prob += (
                    self.Mask[i][j]
                    >= self.E[self.n_target][i][j] + self.c[i] + self.c[j] - 2,
                    f"Mask_LB_{i}_{j}",
                )

        # 2. Permutation Mapping
        for i in range(self.M + 1):
            self.prob += (
                pulp.lpSum([self.p[i][j] for j in range(self.K_nodes)]) == self.c[i],
                f"Perm_Row_{i}",
            )

        for j in range(self.K_nodes):
            self.prob += (
                pulp.lpSum([self.p[i][j] for i in range(self.M + 1)]) == 1,
                f"Perm_Col_{j}",
            )

        # 3. Asymmetric Linearization & Isomorphism check
        for i in range(self.M + 1):
            for j in range(self.K_nodes):

                # Z bounds against Mask matrix
                for k in range(self.M + 1):
                    if k != i:
                        ik_min, ik_max = min(i, k), max(i, k)

                        self.prob += (
                            self.Z[i][k][j] <= self.Mask[ik_min][ik_max],
                            f"Z_Mask_{i}_{k}_{j}",
                        )
                        self.prob += self.Z[i][k][j] <= self.p[k][j], f"Z_p_{i}_{k}_{j}"
                        self.prob += (
                            self.Z[i][k][j]
                            >= self.Mask[ik_min][ik_max] + self.p[k][j] - 1,
                            f"Z_Base_{i}_{k}_{j}",
                        )

                lhs = pulp.lpSum([self.Z[i][k][j] for k in range(self.M + 1) if k != i])

                # Target graph mapping limits
                rhs_1 = pulp.lpSum([self.p[i][l] * self.T[l][j] for l in range(j)])
                rhs_2 = pulp.lpSum(
                    [self.p[i][l] * self.T[j][l] for l in range(j + 1, self.K_nodes)]
                )

                self.prob += lhs == rhs_1 + rhs_2, f"Iso_Match_{i}_{j}"


# ==========================================
# Test Execution Block
# ==========================================
if __name__ == "__main__":
    # Mocking small parameters
    N_val = 6
    M_val = 3

    E_0 = [
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ]
    L_0 = [
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 0],
    ]
    T = [
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ]

    # E_0 = [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]]
    # L_0 = [[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]]
    # E_N = [[0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]]
    model = build_model(N_val, M_val, E_0, T, L_0)

    model.writeLP("quantum_model.lp")

    gurobi_model = gp.read("quantum_model.lp")

    # gurobi_model.optimize()

    # if gurobi_model.status == GRB.INFEASIBLE:
    #     print("\nModel is Infeasible! Computing IIS...")

    #     # This isolates the exact equations causing the paradox
    #     gurobi_model.computeIIS()

    #     # Write the contradictory equations to a new text file
    #     gurobi_model.write("paradox_report.ilp")
    #     print(
    #         "\nSUCCESS: Open 'paradox_report.ilp' to see the exact constraints that contradict each other."
    #     )

    # Solve (Use your preferred solver here: pulp.GUROBI(), pulp.CPLEX_CMD(), etc.)
    # Writes every constraint, variable bound, and objective to a text file
    model.solve(pulp.GUROBI(msg=1))

    # 2. Print the E Matrix over time to spot teleportation
    print("\n" + "=" * 30)
    print("      E MATRIX TRACKER")
    print("=" * 30)
    for n in range(N_val + 2):
        print(f"\n--- Time Step n={n} ---")
        for i in range(M_val + 1):
            row = []
            for j in range(M_val + 1):
                # Safely grab the variable from the PuLP dictionary
                var = model.variablesDict().get(f"E_{n}_{i}_{j}")
                val = int(var.varValue) if var and var.varValue is not None else 0
                row.append(val)
            print(row)

    print("\n" + "=" * 30)
    print("      L MATRIX TRACKER")
    print("=" * 30)
    for n in range(N_val + 2):
        print(f"\n--- Time Step n={n} ---")
        for i in range(M_val + 1):
            row = []
            for j in range(M_val + 1):
                # Safely grab the variable from the PuLP dictionary
                var = model.variablesDict().get(f"L_{n}_{i}_{j}")
                val = int(var.varValue) if var and var.varValue is not None else 0
                row.append(val)
            print(row)

    model.writeLP("quantum_model.lp")
    print("Model written to quantum_model.lp")

    print(f"\n--- STATUS ---")

    print(f"Status: {pulp.LpStatus[model.status]}")
    print(f"Objective Value: {pulp.value(model.objective)}")

    # 1. Print all active 'y' operations
    print("\n" + "=" * 30)
    print("      ACTIVE OPERATIONS")
    print("=" * 30)
    for v in model.variables():
        # Using > 0.5 to safely check binary 1 against floating point inaccuracies
        if (
            v.name.startswith("y_")
            and v.varValue is not None
            and v.varValue > 0.5
            and not ("active" in v.name)  # Exclude active sum variables
        ):
            print(f"{v.name} = 1.0")
