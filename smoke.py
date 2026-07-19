import time, soundfile as sf, mlx_whisper
audio, sr = sf.read("test.wav", dtype="float32")
t0 = time.time()
r = mlx_whisper.transcribe(audio, path_or_hf_repo="mlx-community/whisper-large-v3-turbo", language="ru")
print(f"[{time.time()-t0:.1f}s] {r['text'].strip()}")
