import librosa, librosa.display, matplotlib.pyplot as plt
y, sr = librosa.load("audio.wav", sr=24000)
S = librosa.feature.melspectrogram(y=y, sr=sr)
librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
plt.axis('off'); plt.savefig("dataset/spectrogram_01.png", bbox_inches='tight', pad_inches=0)
