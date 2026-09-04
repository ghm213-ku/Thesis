import pulp


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
        ),
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
    # 4. Global Structural Constraints
    # ==========================================
    for n in range(N + 1):
        # Active Flags Definitions
        prob += (
            y_LC_active[n] == pulp.lpSum([y_LC[n][i] for i in range(M + 1)]),
            f"Def_LC_Active_{n}",
        )
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

        # Operation Sum
        prob += (
            y_dummy[n] + y_LC_active[n] + y_CZ_active[n] + y_F_active[n] == 1,
            f"One_Op_{n}",
        )

        if n < N:
            prob += y_dummy[n + 1] >= y_dummy[n], f"Dummy_Mono_{n}"

        # Parity Locks and Self-Targeting Blocks
        for i in range(M + 1):
            for j in range(M + 1):
                if (i % 2) != (j % 2) or i == j:
                    prob += y_CZ[n][i][j] == 0, f"Parity_CZ_{n}_{i}_{j}"
                    prob += y_F[n][i][j] == 0, f"Parity_F_{n}_{i}_{j}"

    for n in range(N + 2):
        for j in range(M + 1):
            prob += (
                pulp.lpSum([L[n][i][j] for i in range(M + 1)]) == 1,
                f"L_sum_{n}_{j}",
            )
        for i in range(M + 1):
            prob += E[n][i][i] == 0, f"No_Self_Edge_{n}_{i}"
            for j in range(M + 1):
                prob += L[n][i][j] <= L[n][i][i], f"L_diag_{n}_{i}_{j}"

    for n in range(N + 1):
        for i in range(M + 1):
            for j in range(i, M + 1):
                prob += (
                    E[n + 1][i][j] - E[n][i][j] <= 1 - y_dummy[n],
                    f"Dummy_UB_{n}_{i}_{j}",
                )
                prob += (
                    E[n][i][j] - E[n + 1][i][j] <= 1 - y_dummy[n],
                    f"Dummy_LB_{n}_{i}_{j}",
                )

        for i in range(M + 1):
            for j in range(M + 1):
                prob += (
                    L[n + 1][i][j] - L[n][i][j] <= 1 - y_LC_active[n],
                    f"LC_L_UB_{n}_{i}_{j}",
                )
                prob += (
                    L[n][i][j] - L[n + 1][i][j] <= 1 - y_LC_active[n],
                    f"LC_L_LB_{n}_{i}_{j}",
                )

    # ==========================================
    # 5. Operation-Specific Constraints
    # ==========================================
    for n in range(N + 1):

        # -----------------------------------
        # LC Operation
        # -----------------------------------
        for i in range(M + 1):
            for j1 in range(M + 1):
                for j2 in range(j1, M + 1):

                    e_1 = E[n][min(i, j1)][max(i, j1)]
                    e_2 = E[n][min(i, j2)][max(i, j2)]

                    lhs = E[n + 1][j1][j2] + E[n][j1][j2] - 1
                    rhs = 3 - y_LC[n][i] - e_1 - e_2

                    prob += lhs <= rhs, f"LC_IF_UB_{n}_{i}_{j1}_{j2}"
                    prob += lhs >= -rhs, f"LC_IF_LB_{n}_{i}_{j1}_{j2}"

                    rhs_only_if = e_1 + e_2 + 2 * (1 - y_LC[n][i])
                    prob += (
                        2 * (E[n + 1][j1][j2] - E[n][j1][j2]) <= rhs_only_if,
                        f"LC_ONLYIF_UB_{n}_{i}_{j1}_{j2}",
                    )
                    prob += (
                        2 * (E[n][j1][j2] - E[n + 1][j1][j2]) <= rhs_only_if,
                        f"LC_ONLYIF_LB_{n}_{i}_{j1}_{j2}",
                    )

        # -----------------------------------
        # CZ and F Shared L-Matrix Limit
        # -----------------------------------
        for i in range(M + 1):
            for j in range(M + 1):
                for x in range(M + 1):
                    prob += (
                        y_CZ[n][i][j] + y_F[n][i][j] <= 2 - L[n][x][j] - L[n][x][i],
                        f"CZ_F_Limit_{n}_{i}_{j}_{x}",
                    )

        # -----------------------------------
        # CZ and F Base and IF Logic
        # -----------------------------------
        for i in range(M + 1):
            for j in range(i, M + 1):

                prob += (
                    E[n + 1][i][j] >= y_CZ[n][i][j] + y_F[n][i][j],
                    f"E_CZ_F_Base_{n}_{i}_{j}",
                )

                prob += (
                    E[n + 1][i][j] - E[n][i][j] <= y_CZ[n][i][j] + 1 - y_CZ_active[n],
                    f"CZ_E_Inertia_UB_{n}_{i}_{j}",
                )
                prob += (
                    E[n][i][j] - E[n + 1][i][j] <= y_CZ[n][i][j] + 1 - y_CZ_active[n],
                    f"CZ_E_Inertia_LB_{n}_{i}_{j}",
                )

                for x in range(M + 1):
                    for y in range(M + 1):
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

        # -----------------------------------
        # L-Matrix ONLY IF Logic
        # -----------------------------------
        for i in range(M + 1):
            for j in range(M + 1):
                for x in range(M + 1):
                    for y in range(M + 1):
                        # 2x Lower Bound (No z-dependency)
                        rhs_2x = (
                            L[n][x][j]
                            + L[n][x][y]
                            + 2 * (1 - y_CZ[n][i][j] - y_F[n][i][j])
                        )
                        prob += (
                            2 * (L[n][x][y] - L[n + 1][x][y]) <= rhs_2x,
                            f"L_Diff_LB_{n}_{i}_{j}_{x}_{y}",
                        )

                        # 3x Upper Bound (With z-buffer)
                        for z in range(M + 1):
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

        # -----------------------------------
        # F Operation Edge Logic
        # -----------------------------------
        for i in range(M + 1):
            for j in range(i, M + 1):
                for k in range(j, M + 1):

                    prob += (
                        E[n + 1][i][k] >= y_F[n][i][j] + E[n][j][k] - 1,
                        f"F_Ek1_{n}_{i}_{j}_{k}",
                    )
                    prob += (
                        E[n + 1][j][k] <= 2 - y_F[n][i][j] - E[n][j][k],
                        f"F_Ek2_{n}_{i}_{j}_{k}",
                    )

                    rhs_F_only = y_F[n][i][j] + E[n][j][k] + 2 * (1 - y_F[n][i][j])
                    prob += (
                        2 * (E[n + 1][i][k] - E[n][i][k]) <= rhs_F_only,
                        f"F_E_OnlyIf_UB_{n}_{i}_{j}_{k}",
                    )
                    prob += (
                        2 * (E[n][j][k] - E[n + 1][j][k]) <= rhs_F_only,
                        f"F_E_OnlyIf_LB_{n}_{i}_{j}_{k}",
                    )

    return prob


# ==========================================
# Test Execution Block
# ==========================================
if __name__ == "__main__":
    N_val = 10
    M_val = 3

    E_0 = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    L_0 = [[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    E_N = [[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]]

    model = build_model(N_val, M_val, E_0, E_N, L_0)
    # Solve the model
    model.solve()

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
