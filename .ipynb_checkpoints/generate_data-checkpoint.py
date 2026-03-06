import numpy as np

# =======================
# PARAMETRI GLOBALI
# =======================

fs = 200.0
N = 256

band_low = 40.0
band_high = 60.0

num_samples_per_class = 500

# Intervale pentru randomizare (robustețe)
A_min, A_max = 0.2, 1.5          # amplitudine semnal (clasa 1)
sigma_min, sigma_max = 0.5, 2.0  # zgomot (ambele clase)

# Hard negatives: uneori introducem un sinus în afara benzii în clasa 0
hard_neg_prob = 0.30             # probabilitatea să apară un interferer în y=0
A_out_min, A_out_max = 0.2, 1.2  # amplitudine interferer (y=0)
out_low_1, out_high_1 = 5.0, 35.0
out_low_2, out_high_2 = 65.0, 120.0

# RNG cu seed fix (reproductibilitate)
rng = np.random.default_rng(42)

# =======================
# FEATURE EXTRACTION
# =======================

def extract_features(x):
    x1_energy = np.sum(x**2)
    x2_variance = np.var(x)

    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1/fs)
    power_spectrum = np.abs(X)**2

    band_mask = (freqs >= band_low) & (freqs <= band_high)
    x3_bandpower = np.sum(power_spectrum[band_mask])

    return x1_energy, x2_variance, x3_bandpower

# =======================
# GENERARE DATASET
# =======================

rows = []
t = np.arange(N) / fs

# ---------
# CLASA y = 0 (NEGATIVE): noise + uneori interferer în afara benzii
# ---------
for _ in range(num_samples_per_class):
    # random sigma: simulează condiții diferite de zgomot
    sigma_i = rng.uniform(sigma_min, sigma_max)

    # zgomot gaussian
    x = rng.normal(0, sigma_i, size=N)

    # hard negative: uneori adăugăm un sinus în afara benzii 40-60
    if rng.uniform(0, 1) < hard_neg_prob:
        # alegem aleator dacă interferer-ul e sub bandă sau peste bandă
        if rng.uniform(0, 1) < 0.5:
            f_out = rng.uniform(out_low_1, out_high_1)
        else:
            f_out = rng.uniform(out_low_2, out_high_2)

        A_out = rng.uniform(A_out_min, A_out_max)
        phase_out = rng.uniform(0, 2*np.pi)
        x = x + A_out * np.sin(2*np.pi*f_out*t + phase_out)

    x1, x2, x3 = extract_features(x)
    rows.append([x1, x2, x3, 0])

# ---------
# CLASA y = 1 (POSITIVE): sinus în bandă + noise
# ---------
for _ in range(num_samples_per_class):
    # random frecvență în bandă: modelul învață "semnal în 40-60", nu "50 fix"
    f = rng.uniform(band_low, band_high)

    # random amplitudine: semnal mai slab/puternic
    A = rng.uniform(A_min, A_max)

    # random sigma: noise diferit pe exemplu
    sigma_i = rng.uniform(sigma_min, sigma_max)

    # random phase: semnal nealiniat
    phase = rng.uniform(0, 2*np.pi)

    signal = A * np.sin(2*np.pi*f*t + phase)
    noise = rng.normal(0, sigma_i, size=N)

    x = signal + noise
    x1, x2, x3 = extract_features(x)
    rows.append([x1, x2, x3, 1])

rows = np.array(rows)

# Shuffle (ca să nu fie întâi toate 0 apoi toate 1)
rng.shuffle(rows)

np.savetxt(
    "data.csv",
    rows,
    delimiter=",",
    header="x1,x2,x3,y",
    comments="",
    fmt="%.3f"
)

print("Dataset generat: data.csv")
print("Shape:", rows.shape)
print("Pozitive:", int(np.sum(rows[:, -1] == 1)), "| Negative:", int(np.sum(rows[:, -1] == 0)))
