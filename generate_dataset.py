# generate_dataset.py
import os, random, numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt

SR = 24000
DUR = 10.0  # secondes
N_MELS = 128
IMG_SIZE = (5, 5)  # ~512x512 px selon DPI

# --- Synthèse simple de sons virtuels pour créer des spectrogrammes ---
def osc_sine(freq, t): return np.sin(2*np.pi*freq*t)
def limiter(x, t=0.99):
    m = np.max(np.abs(x) + 1e-9)
    return (t/m) * x if m > t else x

def add_noise(x, amt=0.01): return limiter(x + np.random.randn(len(x))*amt)

def env_adsr(x, sr=SR, a=0.02, d=0.1, s=0.7, r=0.2):
    n = len(x)
    a_n, d_n, r_n = int(a*sr), int(d*sr), int(r*sr)
    s_n = max(0, n - (a_n+d_n+r_n))
    env = np.concatenate([
        np.linspace(0, 1, max(1,a_n)),
        np.linspace(1, s, max(1,d_n)),
        np.full(max(1,s_n), s),
        np.linspace(s, 0, max(1,r_n))
    ])
    return x[:len(env)] * env[:len(x)]

# --- Petites “recettes sonores” selon les émotions ---
def synth(freqs, tempo, sr=SR, dur=DUR):
    t = np.linspace(0, dur, int(sr*dur), endpoint=False)
    y = np.zeros_like(t)
    beat = int(sr*60/tempo/4)
    for i in range(0, len(t), beat):
        f = random.choice(freqs)
        seg = env_adsr(osc_sine(f, t[i:i+beat]), sr, a=0.01, d=0.05, s=0.7, r=0.05)
        y[i:i+len(seg)] += seg
    return limiter(add_noise(y, 0.005))

PRESETS = {
    "joie":      {"freqs":[261.63,329.63,392.00,523.25],"tempo":140},
    "tristesse": {"freqs":[220.00,246.94,293.66],"tempo":70},
    "colere":    {"freqs":[82.41,110.00,146.83],"tempo":160},
    "calme":     {"freqs":[110.00,164.81,220.00],"tempo":80},
    "mystere":   {"freqs":[196.00,207.65,233.08],"tempo":100},
    "energie":   {"freqs":[261.63,329.63,392.00,493.88],"tempo":170}
}

# --- Création du spectrogramme ---
def save_spectrogram_png(y, sr, path_png):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    S_db = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=IMG_SIZE)
    librosa.display.specshow(S_db, sr=sr, x_axis=None, y_axis=None, cmap="magma")
    plt.axis("off")
    plt.tight_layout(pad=0)
    os.makedirs(os.path.dirname(path_png), exist_ok=True)
    plt.savefig(path_png, bbox_inches="tight", pad_inches=0)
    plt.close()

# --- Génération du dataset ---
def main(out_dir="spectrogram_dataset", per_class=30):
    os.makedirs(out_dir, exist_ok=True)
    idx = 0
    for emo, params in PRESETS.items():
        for i in range(per_class):
            y = synth(params["freqs"], params["tempo"])
            base = f"{emo}_{idx:04d}"
            png_path = os.path.join(out_dir, base + ".png")
            save_spectrogram_png(y, SR, png_path)
            with open(os.path.join(out_dir, base + ".txt"), "w", encoding="utf-8") as f:
                f.write(f"a {emo} spectrogram image of synthetic audio, 24kHz, 10s, clean texture")
            idx += 1
    print(f"✅ Dataset créé : {out_dir} ({idx} images)")

if __name__ == "__main__":
    main()
