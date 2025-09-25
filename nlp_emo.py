# nlp_emo.py
# Détection d'émotion très légère par lexique (CPU-only)

from collections import defaultdict
import json
import re
import unicodedata
from pathlib import Path

# --- 1 Uniformisation du texte

def normalize(text: str) -> str:
    """
    - passe en minuscules
    - enlève les accents
    - supprime la ponctuation superflue (on garde espaces)
    """
    text = text.lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    # Remplace toute ponctuation par espace
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Compacte espaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- 2 Lexique  ------------------------------------

DEFAULT_LEXICON = {
    "joie": {
        "heureux": 2, "heureuse": 2, "content": 2, "contente": 2, "fete": 1,
        "soleil": 1, "plage": 1, "rire": 2, "rire": 2, "amusement": 1,
        "enthousiasme": 2, "sourire": 1, "joie": 2, "bonheur": 2, "satisfait": 1,
        "amusant": 1, "excite": 2, "motivé": 2, "ravi": 2, "celebration": 1
    },
    "tristesse": {
        "triste": 2, "tristesse": 2, "pluie": 1, "nostalgie": 2, "perdu": 1,
        "solitude": 2, "chagrin": 2, "melancolie": 2, "pleurer": 2,
        "deprime": 2, "desespoir": 2, "fatigue": 1, "ennui": 1,
        "larmes": 2, "coeur brise": 2, "deprimee": 2
    },
    "colere": {
        "colere": 2, "rage": 2, "furieux": 2, "furieuse": 2, "combat": 1,
        "haine": 2, "agressif": 1, "agressive": 1, "enervement": 2,
        "tension": 1, "crise": 1, "violence": 2, "explosion": 1,
        "frustration": 2, "nerveux": 1, "bouleverse": 1
    },
    "calme": {
        "calme": 2, "zen": 2, "paisible": 2, "nuit": 1, "douceur": 1,
        "repos": 1, "silence": 1, "apaisant": 2, "detente": 2,
        "relax": 2, "serenite": 2, "plaisible": 1, "tranquille": 2,
        "reposant": 2, "harmonie": 2
    },
    "mystere": {
        "mystere": 2, "enigme": 2, "suspense": 2, "ombre": 1, "lune": 1,
        "secret": 1, "etrange": 2, "brume": 1, "fantome": 1,
        "inconnu": 1, "creepy": 1, "sombre": 2, "chuchotement": 1,
        "caché": 1, "bizarre": 1
    },
    "energie": {
        "energie": 2, "vitesse": 1, "rapide": 2, "festif": 1, "boom": 1,
        "danse": 1, "puissant": 2, "intense": 2, "sport": 1,
        "dynamique": 2, "adrenaline": 2, "excitante": 2, "mouvement": 1,
        "accelere": 1, "explosif": 2
    }
}


# --- 3 Chargement optionnel d'un lexique JSON ------------------------------

def load_lexicon(path: str | Path | None = None) -> dict:
    """
    Si un JSON est fourni (format: {"emo":{"mot":poids,...}, ...}), on le charge.
    Sinon on retourne le lexique par défaut.
    """
    if path and Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_LEXICON

# --- 4 Scorage & prédiction ------------------------------------------------

def score_emotions(text: str, lexicon: dict) -> dict:
    """
    Renvoie un dict {emotion: score}. Score = somme des poids des mots trouvés.
    - Matching par mots entiers (regex \bmot\b)
    - Texte normalisé pour robustesse
    """
    t = normalize(text)
    scores = defaultdict(int)
    for emo, words in lexicon.items():
        for w, weight in words.items():
            # \b pour mot entier; escape au cas où
            if re.search(rf"\b{re.escape(w)}\b", t):
                scores[emo] += weight
    return dict(scores)

def guess_emotion(text: str, lexicon: dict, default: str = "calme") -> str:
    scores = score_emotions(text, lexicon)
    if not scores:
        return default
    # renvoie l'émotion la mieux scorée ; tie-break: ordre alphabétique stable
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

# --- 5 Démo rapide ---------------------------------------------------------

if __name__ == "__main__":
    lx = load_lexicon()  # ou load_lexicon("lexique.json")
    tests = [
        "Soleil, plage et sourire, je suis super content !",
        "Il pleut, c'est mélancolique, j'ai envie de pleurer.",
        "Je suis furieux, quelle rage !",
        "Ambiance zen, douceur et silence ce soir.",
        "Un secret étrange sous la brume… quel mystère.",
        "Boom boom, c'est intense et rapide, plein d'énergie !",
    ]
    for t in tests:
        emo = guess_emotion(t, lx)
        print(f"[{t}] -> {emo} | scores={score_emotions(t, lx)}")
