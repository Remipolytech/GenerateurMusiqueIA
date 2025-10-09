# generator.py
from music21 import stream, note, tempo, instrument, meter, key
import random, os

# Association émotion → paramètres musicaux
PRESETS = {
    "joie":      {"mode": ("C", "major"),  "bpm": (120, 140)},
    "tristesse": {"mode": ("A", "minor"),  "bpm": (60, 80)},
    "colere":    {"mode": ("E", "minor"),  "bpm": (140, 170)},
    "calme":     {"mode": ("D", "dorian"), "bpm": (70, 90)},
    "mystere":   {"mode": ("D", "dorian"), "bpm": (80, 100)},
    "energie":   {"mode": ("G", "mixolydian"), "bpm": (130, 160)}
}

SCALES = {
    ("C","major"):   ["C4","D4","E4","F4","G4","A4","B4","C5"],
    ("A","minor"):   ["A3","B3","C4","D4","E4","F4","G4","A4"],
    ("E","minor"):   ["E3","F#3","G3","A3","B3","C4","D4","E4"],
    ("D","dorian"):  ["D3","E3","F3","G3","A3","B3","C4","D4"],
    ("G","mixolydian"): ["G3","A3","B3","C4","D4","E4","F4","G4"],
}

def make_melody(emotion: str, bars: int = 4, seed: int | None = None):
    random.seed(seed)
    p = PRESETS.get(emotion, PRESETS["calme"])
    scale = SCALES[p["mode"]]
    bpm = random.randint(*p["bpm"])

    s = stream.Stream()
    s.append(tempo.MetronomeMark(number=bpm))
    s.append(meter.TimeSignature('4/4'))
    s.append(instrument.Piano())
    tonic, mode = p["mode"]
    s.append(key.Key(tonic, mode))

    idx = random.randrange(len(scale))
    for _ in range(bars * 4):  # environ 16 notes
        step = random.choice([-1, 1, 2])
        idx = max(0, min(len(scale)-1, idx + step))
        n = note.Note(scale[idx])
        n.quarterLength = 0.5
        s.append(n)

    return s

def save_melody(emotion: str, outfile="output/test.mid"):
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    melody = make_melody(emotion)
    melody.write("midi", fp=outfile)
    print(f" Mélodie générée pour {emotion} -> {outfile}")

if __name__ == "__main__":
    save_melody("joie")
