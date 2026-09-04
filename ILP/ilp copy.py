import pulp
import gurobipy as gp
from gurobipy import GRB


def build_model(N, M, E_0, E_N_plus_1, L_0):
    prob = pulp.LpProblem("Quantum_Circuit_Routing", pulp.LpMinimize)

    # ==========================================
    # 1. Variables
    # ==========================================
    y_dummy = pulp.LpVariable.dicts("y_dummy", range(N + 1), cat="Binary")
    y_LC = pulp.LpVariable.dicts("y_LC", (range(N + 1), range(M + 1)), cat="Binary")
    y_CZ = pulp.LpVariable.dicts(
        "y_CZ", (range(N + 1), range(M + 1), range(M + 1)), cat="Binary"
    )
    y_F = pulp.LpVariable.dicts(
        "y_F", (range(N + 1), range(M + 1), range(M + 1)), cat="Binary"
    )

    y_LC_active = pulp.LpVariable.dicts("y_LC_active", range(N + 1), cat="Binary")
    y_CZ_active = pulp.LpVariable.dicts("y_CZ_active", range(N + 1), cat="Binary")
    y_F_active = pulp.LpVariable.dicts("y_F_active", range(N + 1), cat="Binary")

    E = pulp.LpVariable.dicts(
        "E", (range(N + 2), range(M + 1), range(M + 1)), cat="Binary"
    )
    L = pulp.LpVariable.dicts(
        "L", (range(N + 2), range(M + 1), range(M + 1)), cat="Binary"
    )

    # ==========================================
    # 2. Objective Function
    # ==========================================
    prob += (
        pulp.lpSum(
            [
                3.17 * y_CZ[n][i][j] + 1.0 * y_F[n][i][j]
                for n in range(N + 1)
                for i in range(M + 1)
                for j in range(M + 1)
            ]
        )
        + pulp.lpSum([0.01 * y_LC[n][i] for n in range(N + 1) for i in range(M + 1)]),
        "Minimize_Cost",
    )

    # ==========================================
    # 3. Boundary Conditions
    # ==========================================
    for i in range(M + 1):
        for j in range(M + 1):
            prob += E[0][i][j] == E_0[i][j], f"Init_E_{i}_{j}"
            prob += L[0][i][j] == L_0[i][j], f"Init_L_{i}_{j}"
            prob += E[N + 1][i][j] == E_N_plus_1[i][j], f"Final_E_{i}_{j}"

    # ==========================================
    # 4. Global Operations & Setup
    # ==========================================
    for n in range(N + 1):

        # Operation Sum = 1
        prob += (
            y_dummy[n]
            + pulp.lpSum([y_LC[n][i] for i in range(M + 1)])
            + pulp.lpSum(
                [
                    y_CZ[n][i][j] + y_F[n][i][j]
                    for i in range(M + 1)
                    for j in range(M + 1)
                ]
            )
            == 1,
            f"One_Op_{n}",
        )

        # Dummy Monotonicity
        if n < N:
            prob += y_dummy[n + 1] >= y_dummy[n], f"Dummy_Mono_{n}"

        # Dummy E-Matrix Inertia
        for i in range(M + 1):
            for j in range(i, M + 1):
                prob += (
                    E[n + 1][i][j] - E[n][i][j] <= 1 - y_dummy[n],
                    f"Dummy_E_UB_{n}_{i}_{j}",
                )
                prob += (
                    E[n][i][j] - E[n + 1][i][j] <= 1 - y_dummy[n],
                    f"Dummy_E_LB_{n}_{i}_{j}",
                )

        # Parity constraints
        for i in range(M + 1):
            for j in range(M + 1):
                # prob += (
                #     y_F[n][i][j] == 0,
                #     f"testing Parity_F_{n}_{i}_{j}",
                # )  # TESTING: Removed F possiblity
                # prob += (
                #     y_CZ[n][i][j] == 0,
                #     f"testing Parity_CZ_{n}_{i}_{j}",
                # )  # TESTING: Removed F possiblity
                if (
                    i % 2 != j % 2 or i == j
                ):  # i and j must be of different parity and not equal
                    prob += y_CZ[n][i][j] == 0, f"Parity_CZ_{n}_{i}_{j}"
                    prob += y_F[n][i][j] == 0, f"Parity_F_{n}_{i}_{j}"

    # DSU & E-Matrix Zeros (Applies to all timesteps)
    for n in range(N + 2):
        for j in range(M + 1):
            # Column sum = 1
            prob += (
                pulp.lpSum([L[n][i][j] for i in range(M + 1)]) == 1,
                f"DSU_Col_Sum_{n}_{j}",
            )

        for i in range(M + 1):
            for j in range(M + 1):
                prob += L[n][i][j] <= L[n][i][i], f"DSU_Diag_{n}_{i}_{j}"

            # E_n[i][j] = 0 for i >= j
            for j in range(i + 1):
                prob += E[n][i][j] == 0, f"E_Zero_Lower_{n}_{i}_{j}"

    # ==========================================
    # 5. LC Operation
    # ==========================================
    for n in range(N + 1):

        # LC Active definition
        prob += (
            y_LC_active[n] == pulp.lpSum([y_LC[n][i] for i in range(M + 1)]),
            f"Def_LC_Active_{n}",
        )

        # LC/Dummy L-Matrix Inertia
        for i in range(M + 1):
            for j in range(M + 1):
                prob += (
                    L[n + 1][i][j] - L[n][i][j] <= 1 - y_LC_active[n] - y_dummy[n],
                    f"LC_Dum_L_UB_{n}_{i}_{j}",
                )
                prob += (
                    L[n][i][j] - L[n + 1][i][j] <= 1 - y_LC_active[n] - y_dummy[n],
                    f"LC_Dum_L_LB_{n}_{i}_{j}",
                )

        for i in range(M + 1):

            # Direct Edge Inertia (for all i, j >= i)
            for j in range(i + 1, M + 1):
                prob += (
                    E[n + 1][i][j] - E[n][i][j] <= 1 - y_LC[n][i] - y_LC[n][j],
                    f"LC_Direct_UB_{n}_{i}_{j}",
                )
                prob += (
                    E[n][i][j] - E[n + 1][i][j] <= 1 - y_LC[n][i] - y_LC[n][j],
                    f"LC_Direct_LB_{n}_{i}_{j}",
                )

            for j1 in range(M + 1):
                for j2 in range(j1 + 1, M + 1):
                    if i != j1 and i != j2 and j1 != j2:  # i != j1 != j2

                        e_1 = E[n][min(i, j1)][max(i, j1)]
                        e_2 = E[n][min(i, j2)][max(i, j2)]

                        # IF logic
                        lhs_if = E[n + 1][j1][j2] + E[n][j1][j2] - 1
                        rhs_if = 3 - y_LC[n][i] - e_1 - e_2

                        prob += lhs_if <= rhs_if, f"LC_IF_UB_{n}_{i}_{j1}_{j2}"
                        prob += lhs_if >= -rhs_if, f"LC_IF_LB_{n}_{i}_{j1}_{j2}"

                        # ONLY IF logic
                        rhs_only = e_1 + e_2 + 2 * (1 - y_LC[n][i])
                        prob += (
                            2 * (E[n + 1][j1][j2] - E[n][j1][j2]) <= rhs_only,
                            f"LC_ONLYIF_UB_{n}_{i}_{j1}_{j2}",
                        )
                        prob += (
                            2 * (E[n][j1][j2] - E[n + 1][j1][j2]) <= rhs_only,
                            f"LC_ONLYIF_LB_{n}_{i}_{j1}_{j2}",
                        )

    # ==========================================
    # 6. CZ and Common Operation Logic
    # ==========================================
    for n in range(N + 1):

        # Active sums for CZ and F
        prob += (
            y_CZ_active[n]
            == pulp.lpSum([y_CZ[n][a][b] for a in range(M + 1) for b in range(M + 1)]),
            f"Def_CZ_Active_{n}",
        )
        prob += (
            y_F_active[n]
            == pulp.lpSum([y_F[n][a][b] for a in range(M + 1) for b in range(M + 1)]),
            f"Def_F_Active_{n}",
        )

        # E-Matrix CZ/F Global Inertia
        for a in range(M + 1):
            for b in range(a + 1, M + 1):
                f_sum = pulp.lpSum(
                    [
                        y_F[n][a][x] + y_F[n][x][a] + y_F[n][b][x] + y_F[n][x][b]
                        for x in range(M + 1)
                    ]
                )
                rhs_inertia = y_CZ[n][a][b] + f_sum + 1 - y_CZ_active[n] - y_F_active[n]

                prob += (
                    E[n + 1][a][b] - E[n][a][b] <= rhs_inertia,
                    f"Global_E_UB_{n}_{a}_{b}",
                )
                prob += (
                    E[n][a][b] - E[n + 1][a][b] <= rhs_inertia,
                    f"Global_E_LB_{n}_{a}_{b}",
                )

        # L-Matrix Bounds and E-Matrix Base for CZ/F
        for i in range(M + 1):
            for j in range(M + 1):

                # y_CZ + y_F <= 2 - L[x][j] - L[x][i] (Applies to all i,j)
                for x in range(M + 1):
                    prob += (
                        y_CZ[n][i][j] + y_F[n][i][j] <= 2 - L[n][x][j] - L[n][x][i],
                        f"CZ_F_Limit_{n}_{i}_{j}_{x}",
                    )

                for x in range(M + 1):
                    for y in range(M + 1):

                        rhs_2x = (
                            L[n][x][j]
                            + L[n][x][y]
                            + 2 * (1 - y_CZ[n][i][j] - y_F[n][i][j])
                        )
                        prob += (
                            2 * (L[n][x][y] - L[n + 1][x][y]) <= rhs_2x,
                            f"L_Diff_LB_{n}_{i}_{j}_{x}_{y}",
                        )

                        for z in range(M + 1):
                            prob += (
                                y_CZ[n][i][j]
                                + y_F[n][i][j]
                                + L[n][x][j]
                                + L[n][x][y]
                                + L[n][z][i]
                                - L[n + 1][z][y]
                                <= 3,
                                f"CZ_F_L1_{n}_{i}_{j}_{x}_{y}_{z}",
                            )
                            prob += (
                                y_CZ[n][i][j]
                                + y_F[n][i][j]
                                + L[n][x][j]
                                + L[n][x][y]
                                + L[n][z][i]
                                + L[n + 1][x][y]
                                <= 4,
                                f"CZ_F_L2_{n}_{i}_{j}_{x}_{y}_{z}",
                            )

                            rhs_3x = (
                                L[n][x][i]
                                + L[n][z][j]
                                + L[n][z][y]
                                + 3 * (1 - y_CZ[n][i][j] - y_F[n][i][j])
                                + 3 * (1 - L[n][z][j])
                            )
                            prob += (
                                3 * (L[n + 1][x][y] - L[n][x][y]) <= rhs_3x,
                                f"L_Diff_UB_{n}_{i}_{j}_{x}_{y}_{z}",
                            )

                # Equations specifically for j >= i
                if j > i:
                    prob += (
                        E[n + 1][i][j]
                        >= y_CZ[n][i][j] + y_F[n][i][j] + y_CZ[n][j][i] + y_F[n][j][i],
                        f"E_CZ_F_Base_{n}_{i}_{j}",
                    )

    # ==========================================
    # 7. F Operation Edge Logic
    # ==========================================
    for n in range(N + 1):
        for i in range(M + 1):
            for j in range(M + 1):
                if i != j:
                    for k in range(M + 1):
                        if k != i and k != j:

                            ik_min, ik_max = min(i, k), max(i, k)
                            jk_min, jk_max = min(j, k), max(j, k)

                            e_ik_n1 = E[n + 1][ik_min][ik_max]
                            e_ik_n = E[n][ik_min][ik_max]
                            e_jk_n1 = E[n + 1][jk_min][jk_max]
                            e_jk_n = E[n][jk_min][jk_max]

                            # IF logic
                            prob += (
                                e_ik_n1 >= y_F[n][i][j] + e_jk_n - 1,
                                f"F_Ek1_{n}_{i}_{j}_{k}",
                            )
                            prob += (
                                e_jk_n1 <= 2 - y_F[n][i][j] - e_jk_n,
                                f"F_Ek2_{n}_{i}_{j}_{k}",
                            )

                            # ONLY IF logic (Upper Ceilings)
                            rhs_f = y_F[n][i][j] + e_jk_n + 2 * (1 - y_F[n][i][j])
                            prob += (
                                2 * (e_ik_n1 - e_ik_n) <= rhs_f,
                                f"F_OnlyIf_UB_{n}_{i}_{j}_{k}",
                            )
                            prob += (
                                2 * (e_jk_n - e_jk_n1) <= rhs_f,
                                f"F_OnlyIf_LB_{n}_{i}_{j}_{k}",
                            )

                            # MISSING ANTI-LEAK BOUNDS (Lower Floors)
                            prob += (
                                e_ik_n - e_ik_n1 <= 1 - y_F[n][i][j],
                                f"F_No_Disappear_{n}_{i}_{j}_{k}",
                            )
                            prob += (
                                e_jk_n1 - e_jk_n <= 1 - y_F[n][i][j],
                                f"F_No_Spawn_{n}_{i}_{j}_{k}",
                            )

    return prob


# ==========================================
# Example usage to verify construction
# ==========================================
if __name__ == "__main__":
    # Mocking small parameters
    N_val = 5
    M_val = 5

    # Creating empty zero matrices for the parameters
    # mock_E0 = [[0 for _ in range(M_val + 1)] for _ in range(M_val + 1)]
    E_0 = [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ]
    L_0 = [
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0],
    ]
    E_N = [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ]
    # mock_L0 = [[0 for _ in range(M_val + 1)] for _ in range(M_val + 1)]
    # mock_EN = [[0 for _ in range(M_val + 1)] for _ in range(M_val + 1)]

    # For a real run, populate mock_E0, mock_L0, and mock_EN accurately.
    # Set L_0 to diagonal identity just for valid mapping
    # for i in range(M_val + 1):
    #     mock_L0[i][i] = 1

    model = build_model(N_val, M_val, E_0, E_N, L_0)

    # 1. Write the model to a file
    model.writeLP("quantum_model.lp")

    # 2. Load that file directly into Gurobi's native engine
    gurobi_model = gp.read("quantum_model.lp")

    # 3. Tell Gurobi to try solving it
    gurobi_model.optimize()

    # 4. If it crashes, force Gurobi to isolate the contradictory constraints
    if gurobi_model.status == GRB.INFEASIBLE:
        print("\nModel is Infeasible! Computing IIS...")

        # This isolates the exact equations causing the paradox
        gurobi_model.computeIIS()

        # Write the contradictory equations to a new text file
        gurobi_model.write("paradox_report.ilp")
        print(
            "\nSUCCESS: Open 'paradox_report.ilp' to see the exact constraints that contradict each other."
        )

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
        if v.name.startswith("y_") and v.varValue is not None and v.varValue > 0.5:
            print(f"{v.name} = 1.0")
