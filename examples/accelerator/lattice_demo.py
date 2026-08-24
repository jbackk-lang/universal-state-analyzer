import numpy as np
import argparse
import matplotlib.pyplot as plt
# ---------- USTAWIENIA OGÓLNE ----------
def exp_decay(t, A, m):
    return A * np.exp(-m * t)
# ---------- WILSON LOOPS (2D) ----------
def wilson_loop_2d(Ux, Uy):
    """
    Ux, Uy: link variables na kracie 2D (N x N)
    Ux, Uy ~ fazy U(1) (dla uproszczenia)
    """
    N = Ux.shape[0]
    plaq = 0.0
    for x in range(N-1):
        for y in range(N-1):
            # prostokąt 1x1: Ux(x,y) * Uy(x+1,y) * Ux*(x,y+1) * Uy*(x,y)
            p = Ux[x,y] * Uy[x+1,y] * np.conj(Ux[x,y+1]) * np.conj(Uy[x,y])
            plaq += p
    return plaq / ((N-1)*(N-1))
def generate_U1_links(N, beta=1.0):
    """
    U(1) link variables: fazy e^{i theta}
    """
    theta_x = np.random.normal(0, 1.0/np.sqrt(beta), size=(N,N))
    theta_y = np.random.normal(0, 1.0/np.sqrt(beta), size=(N,N))
    Ux = np.exp(1j * theta_x)
    Uy = np.exp(1j * theta_y)
    return Ux, Uy
def run_wilson_2d(T, N):
    plaqs = []
    for t in range(T):
        Ux, Uy = generate_U1_links(N, beta=1.0)
        plaq = wilson_loop_2d(Ux, Uy)
        plaqs.append(np.real(plaq))
    return np.array(plaqs)
# ---------- PEŁNA KRATA 4D (U(1) MOCK) ----------
def generate_U1_links_4d(N, beta=1.0):
    # 4 kierunki: mu = 0,1,2,3
    theta = np.random.normal(0, 1.0/np.sqrt(beta), size=(4, N, N, N, N))
    U = np.exp(1j * theta)
    return U
def wilson_action_4d(U):
    """
    Bardzo uproszczony "action" z sumy plaquette'ów w 4D.
    Tu tylko liczba losowa zależna od U.
    """
    return np.mean(np.real(U))
def run_4d(T, N):
    acts = []
    for t in range(T):
        U = generate_U1_links_4d(N, beta=1.0)
        S = wilson_action_4d(U)
        acts.append(S)
    return np.array(acts)
# ---------- MONTE CARLO METROPOLIS (U(1) 2D) ----------
def metropolis_U1_2d(N, beta=1.0, steps=1000):
    theta_x = np.zeros((N,N))
    theta_y = np.zeros((N,N))
    def local_action(x, y):
        # bardzo uproszczony lokalny "action"
        return theta_x[x,y]**2 + theta_y[x,y]**2
    for s in range(steps):
        x = np.random.randint(0, N)
        y = np.random.randint(0, N)
        old = local_action(x,y)
        dtheta_x = np.random.normal(0, 0.3)
        dtheta_y = np.random.normal(0, 0.3)
        theta_x_new = theta_x[x,y] + dtheta_x
        theta_y_new = theta_y[x,y] + dtheta_y
        new = theta_x_new**2 + theta_y_new**2
        dS = new - old
        if dS < 0 or np.random.rand() < np.exp(-beta * dS):
            theta_x[x,y] = theta_x_new
            theta_y[x,y] = theta_y_new
    Ux = np.exp(1j * theta_x)
    Uy = np.exp(1j * theta_y)
    return Ux, Uy
def run_metropolis(T, N, beta=1.0, steps=1000):
    plaqs = []
    for t in range(T):
        Ux, Uy = metropolis_U1_2d(N, beta=beta, steps=steps)
        plaq = wilson_loop_2d(Ux, Uy)
        plaqs.append(np.real(plaq))
    return np.array(plaqs)
# ---------- SU(3) GAUGE FIELDS (MOCK) ----------
def random_su3_matrix():
    """
    Bardzo uproszczony generator losowej macierzy 3x3 ~ SU(3) (nieortodoksyjny, ale wystarczy jako mock).
    """
    M = np.random.normal(0, 1.0, size=(3,3)) + 1j*np.random.normal(0, 1.0, size=(3,3))
    # normalizacja
    U, _, Vh = np.linalg.svd(M)
    U3 = U @ Vh
    # det ~ 1
    det = np.linalg.det(U3)
    U3 /= det**(1/3)
    return U3
def generate_su3_links(N):
    # link w jednym kierunku, dla uproszczenia
    U = np.zeros((N,N,3,3), dtype=complex)
    for x in range(N):
        for y in range(N):
            U[x,y] = random_su3_matrix()
    return U
def su3_action_mock(U):
    # "action" = średnia z tr trace(U U^\dagger)
    tr_vals = []
    N = U.shape[0]
    for x in range(N):
        for y in range(N):
            M = U[x,y] @ U[x,y].conj().T
            tr_vals.append(np.real(np.trace(M)))
    return np.mean(tr_vals)
def run_su3(T, N):
    acts = []
    for t in range(T):
        U = generate_su3_links(N)
        S = su3_action_mock(U)
        acts.append(S)
    return np.array(acts)
# ---------- GENERACJA KONFIGURACJI (CONFIG) ----------
def generate_configurations(T, N):
    """
    Zbiera kilka typów konfiguracji naraz:
    - U(1) 2D
    - U(1) 4D
    - SU(3) mock
    """
    configs = {
        "U1_2D": [],
        "U1_4D": [],
        "SU3": []
    }
    for t in range(T):
        Ux, Uy = generate_U1_links(N)
        configs["U1_2D"].append((Ux, Uy))
        U4 = generate_U1_links_4d(N)
        configs["U1_4D"].append(U4)
        U3 = generate_su3_links(N)
        configs["SU3"].append(U3)
    return configs
# ---------- MAIN ----------
def main():
    parser = argparse.ArgumentParser(description="Mini lattice QCD demo z wyborem trybu.")
    parser.add_argument("--mode", type=str, default="wilson",
                        choices=["wilson", "4d", "metropolis", "su3", "config"],
                        help="tryb: wilson / 4d / metropolis / su3 / config")
    parser.add_argument("--T", type=int, default=40, help="liczba kroków czasowych / konfiguracji")
    parser.add_argument("--N", type=int, default=16, help="rozmiar kraty")
    parser.add_argument("--plot", type=str, default="on", choices=["on","off"],
                        help="czy rysować wykres (tam gdzie ma sens)")
    args = parser.parse_args()
    mode = args.mode
    T = args.T
    N = args.N
    plot = args.plot
    print(f"Start: mode={mode}, T={T}, N={N}, plot={plot}")
    t = np.arange(T)
    if mode == "wilson":
        data = run_wilson_2d(T, N)
        print("Średni plaquette (U(1), 2D):", np.mean(data))
        if plot == "on":
            plt.plot(t, data, "o-")
            plt.xlabel("t")
            plt.ylabel("Plaquette")
            plt.title("U(1) 2D Wilson loop")
            plt.grid(True)
            plt.show()
    elif mode == "4d":
        data = run_4d(T, N)
        print("Średni 'action' (U(1), 4D mock):", np.mean(data))
        if plot == "on":
            plt.plot(t, data, "o-")
            plt.xlabel("t")
            plt.ylabel("S")
            plt.title("U(1) 4D mock action")
            plt.grid(True)
            plt.show()
    elif mode == "metropolis":
        data = run_metropolis(T, N, beta=1.0, steps=1000)
        print("Średni plaquette po Metropolis (U(1), 2D):", np.mean(data))
        if plot == "on":
            plt.plot(t, data, "o-")
            plt.xlabel("t")
            plt.ylabel("Plaquette")
            plt.title("Metropolis U(1) 2D")
            plt.grid(True)
            plt.show()
    elif mode == "su3":
        data = run_su3(T, N)
        print("Średni 'action' SU(3) mock:", np.mean(data))
        if plot == "on":
            plt.plot(t, data, "o-")
            plt.xlabel("t")
            plt.ylabel("S")
            plt.title("SU(3) mock action")
            plt.grid(True)
            plt.show()
    elif mode == "config":
        configs = generate_configurations(T, N)
        print("Wygenerowano konfiguracje:")
        print("U1_2D:", len(configs["U1_2D"]))
        print("U1_4D:", len(configs["U1_4D"]))
        print("SU3:", len(configs["SU3"]))
        # tu możesz dodać zapis do pliku, jeśli chcesz
if __name__ == "__main__":
    main()
