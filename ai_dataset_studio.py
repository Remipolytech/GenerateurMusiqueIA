# -*- coding: utf-8 -*-
"""
Fine couche PyQt5 qui UTILISE tes modules existants:
- nlp_emo.py : load_lexicon(), analyze_text_emotion(text, lex), emotion_to_prompt(text, emo)
- run.py     : make_prompt(text) [optionnel]
- generator.py : generate_spectrogram(prompt, seed=..., size=(w,h)) -> PIL.Image  [recommandé]
Si un de ces éléments manque, on bascule en fallback minimal pour que l'UI reste utilisable.
"""

from __future__ import annotations
import sys, re, time, json, traceback
from pathlib import Path
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

# ===== PIL / Pandas =====
from PIL import Image
import pandas as pd

# ===== Import de tes modules, avec fallbacks =====
def _safe_import():
    pipe = {}
    try:
        import importlib
        nlp_emo = importlib.import_module("nlp_emo")
        pipe["load_lexicon"] = getattr(nlp_emo, "load_lexicon", None)
        pipe["analyze_text_emotion"] = getattr(nlp_emo, "analyze_text_emotion", None)
        pipe["emotion_to_prompt"] = getattr(nlp_emo, "emotion_to_prompt", None)
    except Exception as e:
        print("[WARN] nlp_emo absent/incomplet:", e)
        # fallbacks simples
        BASIC_LEXICON = {
            "joy": ["joy","happy","joie","heureux"],
            "sad": ["sad","triste"],
            "anger": ["anger","colère","énervé"],
            "fear": ["fear","peur","anxious"],
            "surprise":["surprise","étonné","wow"],
            "disgust": ["disgust","dégoût"],
            "calm": ["calm","calme","zen"]
        }
        class Emo:
            def __init__(self, probs, va=(0.0,0.0)):
                self.probs = probs; self.va = va
        def load_lexicon(): return BASIC_LEXICON
        def analyze_text_emotion(text, lex):
            t = text.lower(); scores = {k:0.0 for k in lex}
            for k,keys in lex.items(): scores[k] = float(sum(kw in t for kw in keys))
            s = sum(scores.values()) or 1.0
            probs = {k:v/s for k,v in scores.items()}
            v = probs.get("joy",0)+probs.get("calm",0) - (probs.get("anger",0)+probs.get("fear",0)+probs.get("sad",0)+probs.get("disgust",0))
            a = probs.get("surprise",0)+probs.get("anger",0) - probs.get("calm",0)
            v = max(-1,min(1,v)); a = max(-1,min(1,a))
            return Emo(probs,(v,a))
        def emotion_to_prompt(text, emo):
            best = max(emo.probs, key=emo.probs.get)
            return f"[emotion:{best}] {text}"
        pipe = dict(load_lexicon=load_lexicon,
                    analyze_text_emotion=analyze_text_emotion,
                    emotion_to_prompt=emotion_to_prompt)

    # run.make_prompt (optionnel)
    try:
        import importlib
        run = importlib.import_module("run")
        pipe["make_prompt"] = getattr(run, "make_prompt", None)
    except Exception as e:
        print("[INFO] run.make_prompt non trouvé :", e)
        pipe["make_prompt"] = None

    # generator.generate_spectrogram (recommandé pour image réelle)
    try:
        import importlib
        gen = importlib.import_module("generator")
        pipe["generate_spectrogram"] = getattr(gen, "generate_spectrogram", None)
    except Exception as e:
        print("[INFO] generator.generate_spectrogram non trouvé :", e)
        pipe["generate_spectrogram"] = None

    return pipe

PIPE = _safe_import()
LEX  = PIPE["load_lexicon"]() if PIPE["load_lexicon"] else {}

# Fallback image (si pas de generator)
def _fallback_image(prompt: str, seed: int = 0, size: Tuple[int,int]=(896,448)) -> Image.Image:
    from PIL import Image, ImageDraw
    import random
    random.seed(seed)
    w,h = size; img = Image.new("RGB",(w,h),(16,16,18)); drw = ImageDraw.Draw(img)
    for x in range(0,w,7):
        v = 45 + int(180*abs(random.random()))
        drw.line([(x,0),(x,h)], fill=(v,v//2,60))
    for _ in range(4000):
        x,y = random.randrange(w), random.randrange(h)
        c = 70 + random.randrange(150)
        img.putpixel((x,y),(c,c,c))
    drw.text((10,10), f"seed={seed}", fill=(220,220,220))
    return img

# ===== Sauvegarde dataset =====
@dataclass
class SaveResult:
    png: Path
    txt: Path
    index_csv: Optional[Path]

def _next_idx(out_dir: Path, emotion: str) -> int:
    import re
    pat = re.compile(rf"^{re.escape(emotion)}_(\d+)\.png$", re.I)
    nmax = 0
    for p in out_dir.glob(f"{emotion}_*.png"):
        m = pat.match(p.name)
        if m: nmax = max(nmax, int(m.group(1)))
    return nmax + 1

def save_pair(out_dir: Path, emotion: str, phrase: str, prompt: str, seed: int, img: Image.Image) -> SaveResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = _next_idx(out_dir, emotion)
    stem = f"{emotion}_{n:05d}"
    png = out_dir / f"{stem}.png"
    txt = out_dir / f"{stem}.txt"
    img.save(png)
    meta = {
        "emotion": emotion, "n": n, "timestamp": int(time.time()),
        "seed": seed, "phrase": phrase, "prompt": prompt,
        "spec": {"width": img.width, "height": img.height}
    }
    txt.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    idx = out_dir / "index.csv"
    row = {
        "filename_png": png.name, "filename_txt": txt.name, "emotion": emotion,
        "n": n, "seed": seed, "phrase": phrase, "prompt": prompt, "timestamp": meta["timestamp"]
    }
    if idx.exists():
        df = pd.read_csv(idx)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(idx, index=False)
    return SaveResult(png, txt, idx)

# ===== PyQt5 UI (sobre/pro) =====
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QTextEdit, QLineEdit, QPushButton, QFileDialog, QComboBox, QSpinBox,
    QHBoxLayout, QVBoxLayout, QGroupBox, QMessageBox, QTableWidget, QTableWidgetItem, QFrame
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    img = img.convert("RGBA")
    w,h = img.size
    data = img.tobytes("raw", "RGBA")
    qimg = QImage(data, w, h, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg)

class Main(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Builder — (branché sur tes modules)")
        self.resize(1080, 680)
        self._last_img: Optional[Image.Image] = None
        self._apply_dark()

        # Ligne dossier
        self.out_dir = QLineEdit(str(Path("dataset").resolve()))
        b_browse = QPushButton("Parcourir…"); b_browse.clicked.connect(self.on_browse)
        row_dir = QHBoxLayout()
        row_dir.addWidget(QLabel("Dossier dataset :"))
        row_dir.addWidget(self.out_dir, 1); row_dir.addWidget(b_browse)

        # Phrase
        self.txt_phrase = QTextEdit(); self.txt_phrase.setPlaceholderText("Tape la phrase utilisateur…")
        self.txt_phrase.setFixedHeight(90)

        # Bouton analyse
        b_analyze = QPushButton("Analyser & Générer le prompt"); b_analyze.clicked.connect(self.on_analyze)

        # Prompt
        self.txt_prompt = QTextEdit(); self.txt_prompt.setPlaceholderText("Prompt généré ici (modifiable).")
        self.txt_prompt.setFixedHeight(120)

        # Emotion + seed
        self.cmb_emotion = QComboBox(); self.cmb_emotion.addItems(["joy","sad","anger","fear","surprise","disgust","calm"])
        self.chk_auto = QComboBox(); self.chk_auto.addItems(["Auto (dominante)", "Manuel"])
        self.chk_auto.setCurrentIndex(0)
        self.spin_seed = QSpinBox(); self.spin_seed.setRange(0, 10_000_000); self.spin_seed.setValue(0)

        row_e = QHBoxLayout()
        row_e.addWidget(QLabel("Mode émotion :")); row_e.addWidget(self.chk_auto)
        row_e.addWidget(QLabel("Émotion :")); row_e.addWidget(self.cmb_emotion)
        row_e.addStretch()
        row_e.addWidget(QLabel("Seed :")); row_e.addWidget(self.spin_seed)

        # Probs table
        self.tbl = QTableWidget(7, 2)
        self.tbl.setHorizontalHeaderLabels(["Émotion","Proba"])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.horizontalHeader().setStretchLastSection(True)
        for i,e in enumerate(["joy","sad","anger","fear","surprise","disgust","calm"]):
            self.tbl.setItem(i,0,QTableWidgetItem(e))
            self.tbl.setItem(i,1,QTableWidgetItem("-"))
        self.tbl.setFixedHeight(210)

        # Aperçu
        self.lbl_prev = QLabel("Aperçu spectrogramme"); self.lbl_prev.setAlignment(Qt.AlignCenter)
        self.lbl_prev.setStyleSheet("border:1px solid #2a2a30; min-height:260px;")

        # Boutons génération/sauvegarde
        b_preview = QPushButton("Aperçu (utilise generator.generate_spectrogram)"); b_preview.clicked.connect(self.on_preview)
        b_save = QPushButton("Enregistrer (png + txt + index.csv)"); b_save.clicked.connect(self.on_save)

        # Layout
        root = QVBoxLayout()
        root.addLayout(row_dir)
        root.addWidget(QLabel("Phrase :")); root.addWidget(self.txt_phrase)
        root.addWidget(b_analyze)
        root.addWidget(QLabel("Prompt :")); root.addWidget(self.txt_prompt)
        root.addLayout(row_e)
        root.addWidget(QLabel("Probabilités d'émotions :")); root.addWidget(self.tbl)
        root.addWidget(self.lbl_prev, 1)
        row_b = QHBoxLayout(); row_b.addWidget(b_preview); row_b.addWidget(b_save); root.addLayout(row_b)
        self.setLayout(root)

    def _apply_dark(self):
        self.setStyleSheet("""
            QWidget{background:#0f0f12;color:#e6e6ee;font-size:14px;}
            QTextEdit,QLineEdit,QComboBox,QSpinBox,QTableWidget{background:#141416;border:1px solid #2a2a30;border-radius:6px;color:#e6e6ee;}
            QPushButton{background:#1b1b20;border:1px solid #2f2f36;padding:8px 12px;border-radius:8px;}
            QPushButton:hover{background:#24242a;}
            QHeaderView::section{background:#16161a;border:1px solid #2a2a30;padding:4px;}
        """)

    def on_browse(self):
        d = QFileDialog.getExistingDirectory(self, "Choisir le dossier dataset", self.out_dir.text())
        if d: self.out_dir.setText(d)

    def on_analyze(self):
        text = self.txt_phrase.toPlainText().strip()
        if not text:
            QMessageBox.warning(self,"Manquant","Saisis une phrase.")
            return
        try:
            emo = PIPE["analyze_text_emotion"](text, LEX) if PIPE["analyze_text_emotion"] else None
            # prompt: priorité à ton run.make_prompt si présent; sinon nlp_emo.emotion_to_prompt
            if PIPE["make_prompt"]:
                prompt = PIPE["make_prompt"](text)
            else:
                prompt = PIPE["emotion_to_prompt"](text, emo) if PIPE["emotion_to_prompt"] else text
            # maj table proba
            order = ["joy","sad","anger","fear","surprise","disgust","calm"]
            if emo and getattr(emo, "probs", None):
                for i,k in enumerate(order):
                    p = float(emo.probs.get(k,0.0))
                    self.tbl.setItem(i,1,QTableWidgetItem(f"{p:.3f}"))
                if self.chk_auto.currentIndex()==0:
                    best = max(emo.probs, key=emo.probs.get)
                    self.cmb_emotion.setCurrentText(best)
            self.txt_prompt.setPlainText(f"[emotion:{self.cmb_emotion.currentText()}] {prompt}")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Erreur analyse", str(e))

    def on_preview(self):
        phrase = self.txt_phrase.toPlainText().strip()
        if not phrase:
            QMessageBox.warning(self,"Manquant","Saisis une phrase.")
            return
        prompt = self.txt_prompt.toPlainText().strip()
        if not prompt:
            self.on_analyze(); prompt = self.txt_prompt.toPlainText().strip()
        seed = self.spin_seed.value()
        try:
            gen = PIPE["generate_spectrogram"]
            if gen is None:
                img = _fallback_image(prompt, seed=seed)
            else:
                img = gen(prompt, seed=seed, size=(896,448))  # utilise TON generator.py
            self._last_img = img
            self.lbl_prev.setPixmap(pil_to_qpixmap(img).scaled(self.lbl_prev.width(), self.lbl_prev.height(),
                                                               Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self,"Erreur génération",str(e))

    def on_save(self):
        from PyQt5.QtWidgets import QMessageBox
        phrase = self.txt_phrase.toPlainText().strip()
        prompt = self.txt_prompt.toPlainText().strip()
        if not phrase or not prompt:
            QMessageBox.warning(self,"Manquant","Génère d’abord le prompt/aperçu.")
            return
        if self._last_img is None:
            self.on_preview()
            if self._last_img is None:
                QMessageBox.warning(self,"Erreur","Impossible de générer l’image.")
                return
        emotion = self.cmb_emotion.currentText()
        seed = self.spin_seed.value()
        out_dir = Path(self.out_dir.text())
        try:
            res = save_pair(out_dir, emotion, phrase, prompt, seed, self._last_img)
            QMessageBox.information(self,"OK", f"Enregistré :\n{res.png.name}\n{res.txt.name}\n(index.csv mis à jour)")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self,"Erreur sauvegarde",str(e))

# ===== Entrée =====
from PyQt5.QtWidgets import QFileDialog, QMessageBox
def main():
    app = QApplication(sys.argv)
    w = Main()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
