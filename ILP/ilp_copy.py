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

    def build(self, isomorphism=True):
        """Master build method. Acts as the table of contents for the model."""
        self._define_variables()
        self._define_objective()

        self._add_boundary_conditions()
        self._add_global_operations()
        self._add_lc_constraints()
        self._add_cz_and_f_shared_logic()
        self._add_f_edge_logic()
        self._add_isomorphism_and_deletion()
        if isomorphism:
            self._add_isomorphism_and_deletion()
        else:
            self._add_target_graph()

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

    def _add_target_graph(self):
        # 1. Enforce the target graph at step N+1
        for i in range(self.K_nodes):
            for j in range(i + 1, self.K_nodes):  # strictly upper triangular
                self.prob += (
                    self.E[self.n_target][i][j] == self.T[i][j],
                    f"Target_Edge_{i}_{j}",
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

    def _add_lc_constraints(self):
        for n in range(self.N + 1):
            for i in range(self.M + 1):

                # Freeze edges directly connected to LC target
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

                for j1 in range(self.M + 1):
                    for j2 in range(j1 + 1, self.M + 1):
                        if i != j1 and i != j2:

                            e1 = self.E[n][min(i, j1)][max(i, j1)]
                            e2 = self.E[n][min(i, j2)][max(i, j2)]

                            # LC 'IF' Flip
                            lhs_if = self.E[n + 1][j1][j2] + self.E[n][j1][j2] - 1
                            rhs_if = 3 - self.y_LC[n][i] - e1 - e2

                            self.prob += lhs_if <= rhs_if, f"LC_IF_UB_{n}_{i}_{j1}_{j2}"
                            self.prob += (
                                lhs_if >= -rhs_if,
                                f"LC_IF_LB_{n}_{i}_{j1}_{j2}",
                            )

                            # LC 'ONLY IF' Flip
                            rhs_only = e1 + e2 + 2 * (1 - self.y_LC[n][i])

                            self.prob += (
                                2 * (self.E[n + 1][j1][j2] - self.E[n][j1][j2])
                                <= rhs_only,
                                f"LC_OnlyIf_UB_{n}_{i}_{j1}_{j2}",
                            )
                            self.prob += (
                                2 * (self.E[n][j1][j2] - self.E[n + 1][j1][j2])
                                <= rhs_only,
                                f"LC_OnlyIf_LB_{n}_{i}_{j1}_{j2}",
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

            # Conservation of Leaders
            self.prob += (
                pulp.lpSum(
                    [self.L[n][x][x] - self.L[n + 1][x][x] for x in range(self.M + 1)]
                )
                == self.y_CZ_active[n] + self.y_F_active[n],
                f"Lead_Kill_{n}",
            )

            for x in range(self.M + 1):
                for y in range(self.M + 1):
                    # DSU Monotonicity (Crucial to prevent random leader swaps)
                    self.prob += (
                        self.L[n][x][y] - self.L[n + 1][x][y]
                        <= self.L[n][x][x] - self.L[n + 1][x][x],
                        f"DSU_Mono_{n}_{x}_{y}",
                    )

                    for z in range(y + 1, self.M + 1):
                        # Edge forces same leader
                        self.prob += (
                            self.L[n][x][y] - self.L[n][x][z] <= 1 - self.E[n][y][z],
                            f"Edge_L_UB_{n}_{x}_{y}_{z}",
                        )
                        self.prob += (
                            self.L[n][x][z] - self.L[n][x][y] <= 1 - self.E[n][y][z],
                            f"Edge_L_LB_{n}_{x}_{y}_{z}",
                        )

            for i in range(self.M + 1):
                for j in range(self.M + 1):
                    # Prevent CZ/F gates on nodes in the same component
                    for x in range(self.M + 1):
                        self.prob += (
                            self.y_CZ[n][i][j] + self.y_F[n][i][j]
                            <= 2 - self.L[n][x][j] - self.L[n][x][i],
                            f"CZ_F_Deny_{n}_{i}_{j}_{x}",
                        )

                    if i != j:
                        # Base Edge Formation
                        self.prob += (
                            self.E[n + 1][min(i, j)][max(i, j)]
                            >= self.y_CZ[n][i][j] + self.y_F[n][i][j],
                            f"E_CZ_F_Base_{n}_{i}_{j}",
                        )

            # Global E-Matrix CZ/F Inertia
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
            for i in range(self.M + 1):
                for j in range(self.M + 1):
                    if i != j:
                        for k in range(self.M + 1):
                            if k != i and k != j:

                                ik_min, ik_max = min(i, k), max(i, k)
                                jk_min, jk_max = min(j, k), max(j, k)

                                e_ik_n1 = self.E[n + 1][ik_min][ik_max]
                                e_ik_n = self.E[n][ik_min][ik_max]
                                e_jk_n1 = self.E[n + 1][jk_min][jk_max]
                                e_jk_n = self.E[n][jk_min][jk_max]

                                yF = self.y_F[n][i][j]

                                # F 'IF' rules
                                self.prob += (
                                    e_ik_n1 >= yF + e_jk_n - 1,
                                    f"F_Ek1_{n}_{i}_{j}_{k}",
                                )
                                self.prob += (
                                    e_jk_n1 <= 2 - yF - e_jk_n,
                                    f"F_Ek2_{n}_{i}_{j}_{k}",
                                )

                                # F 'ONLY IF' rules
                                rhs_only = yF + e_jk_n + 2 * (1 - yF)
                                self.prob += (
                                    2 * (e_ik_n1 - e_ik_n) <= rhs_only,
                                    f"F_OnlyUB_{n}_{i}_{j}_{k}",
                                )
                                self.prob += (
                                    2 * (e_jk_n - e_jk_n1) <= rhs_only,
                                    f"F_OnlyLB_{n}_{i}_{j}_{k}",
                                )

                                # Anti-Leaks
                                self.prob += (
                                    e_ik_n - e_ik_n1 <= 1 - yF,
                                    f"F_Anti1_{n}_{i}_{j}_{k}",
                                )
                                self.prob += (
                                    e_jk_n1 <= 1 - yF,
                                    f"F_Anti2_{n}_{i}_{j}_{k}",
                                )

    def _add_isomorphism_and_deletion(self):

        # 1. Final Mask Logic
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
