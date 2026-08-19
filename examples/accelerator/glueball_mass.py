import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import argparse
# --- operator "glueballowy" (plaquette-like, uproszczony) ---
def glueball_operator(field):
    # tu można podmienić na bardziej złożony operator
    return np.mean(field**2)
# --- różne "nakładki" na pole gluonowe ---
def generate_field_basic(N, t):
    return np.random.normal(0, 0.2, size=(N, N))
def generate_field_noise(N, t):
    # rosnący szum w czasie
    noise = 0.2 + 0.01 * t
    return np.random.normal(0, noise, size=(N, N))
def generate_field_su2_mock(N, t):
    # udawane SU(2): losowe macierze 2x2 z determinantą ~1
    # tu tylko symbolicznie, żeby mieć inną "nakładkę"
    field = np.random.normal(0, 0.2, size=(N, N, 2, 2))
    # normalizacja determinanty w przybliżeniu
    det = field[...,0,0]*field[...,1,1] - field[...,0,1]*field[...,1,0]
    det[det == 0] = 1.0
    field /= np.abs(det)[...,None,None]
    # sprowadzenie do skalarnego pola
    return np.mean(field, axis=(2,3))
# --- korelacja czasowa operatora ---
def compute_correlation(T, N, mode):
    corr = []
    for t in range(T):
        if mode == "basic":
            field = generate_field_basic(N, t)
        elif mode == "noise":
            field = generate_field_noise(N, t)
        elif mode == "su2_mock":
            field = generate_field_su2_mock(N, t)
        else:
            raise ValueError(f"Nieznana nakładka: {mode}")
        corr.append(glueball_operator(field))
    return np.array(corr)
# --- dopasowanie wykładnicze: C(t) = A * exp(-m * t) ---
def exp_decay(t, A, m):
    return A * np.exp(-m * t)
def fit_mass(t, corr):
    # proste dopasowanie, bez bajerów
    popt, pcov = curve_fit(exp_decay, t, corr, p0=(corr[0], 0.1))
    A_fit, m_fit = popt
    return A_fit, m_fit
def main():
    parser = argparse.ArgumentParser(description="Uproszczona symulacja glueballa z wyborem nakładki.")
    parser.add_argument("--T", type=int, default=40, help="liczba kroków czasowych (T)")
    parser.add_argument("--N", type=int, default=32, help="rozmiar kraty (N x N)")
    parser.add_argument("--mode", type=str, default="basic",
                        choices=["basic", "noise", "su2_mock"],
                        help="nakładka: basic / noise / su2_mock")
    parser.add_argument("--plot", type=str, default="on",
                        choices=["on", "off"],
                        help="czy rysować wykres: on / off")
    args = parser.parse_args()
    T = args.T
    N = args.N
    mode = args.mode
    plot_mode = args.plot
    print(f"Start symulacji: T={T}, N={N}, nakładka={mode}, wykres={plot_mode}")
    t = np.arange(T)
    corr = compute_correlation(T, N, mode)
    A_fit, m_fit = fit_mass(t, corr)
    print(f"Dopasowany parametr A: {A_fit:.6f}")
    print(f"Dopasowana masa glueballa (m): {m_fit:.6f}")
    if plot_mode == "on":
        plt.figure(figsize=(8,5))
        plt.plot(t, corr, "o", label="korelacja C(t)")
        plt.plot(t, exp_decay(t, A_fit, m_fit), "-", label=f"dopasowanie: m={m_fit:.4f}")
        plt.xlabel("t")
        plt.ylabel("C(t)")
        plt.title(f"Glueball: nakładka={mode}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
if __name__ == "__main__":
    main()
