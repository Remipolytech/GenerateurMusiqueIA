#!/usr/bin/env python3
# run.py — prompt interactif pour Analyse du texte → Émotion

from nlp_emo import load_lexicon, analyze_text_emotion, emotion_to_prompt

BANNER = """\
=== Texte → Émotion (interactive) ===
Tape ton texte puis Entrée.
Commandes : /quit pour quitter, /help pour l’aide.
"""

HELP = """\
Commandes disponibles :
  /quit        Quitter
  /help        Afficher cette aide
"""

def main():
    lex = load_lexicon()
    print(BANNER)
    while True:
        try:
            text = input("Texte > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not text:
            continue
        if text.startswith("/"):
            if text == "/quit":
                print("Bye!")
                break
            elif text == "/help":
                print(HELP)
            else:
                print("Commande inconnue. /help pour l’aide.")
            continue

        emo = analyze_text_emotion(text, lex)
        prompt = emotion_to_prompt(text, emo)

        print("— Résultat —")
        print("Labels:", emo.labels)
        print("Probs :", {k: round(v, 3) for k, v in emo.probs.items()})
        v, a = emo.va
        print("V/A   :", (round(v, 2), round(a, 2)))
        print("Prompt:", prompt)
        print()

if __name__ == "__main__":
    main()
