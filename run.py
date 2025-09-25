# run.py
from nlp_emo import load_lexicon, guess_emotion
from generator import save_melody
import sys

if __name__ == "__main__":
    user_text = " ".join(sys.argv[1:]) or "Soleil et plage, je suis content"
    lex = load_lexicon()
    emo = guess_emotion(user_text, lex)
    print(f"Texte: {user_text}\nÉmotion détectée: {emo}")
    save_melody(emo, outfile="output/from_text.mid")
