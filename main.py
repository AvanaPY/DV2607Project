import whisper.whisper as whisper
import torch
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from scipy.io.wavfile import write

graphpng = "mygraph.png"        # The file name of the Log-Mel spetrogram image to write to
audio_file = "qtpie.mp3"        # The file name of the audio file to play
model_name = "medium"           # Which model to use, see below and https://github.com/openai/whisper for more information
#   tiny, 
#   base, 
#   small, 
#   medium, 
#   large <-- Too big for my system
#   .en extension for english-only version

# Initialize the model
model = whisper.load_model(model_name)
print(f'Running model on device: {model.device}')

# Load the audio and pad or trim it to 30 seconds, as expected by Whisper
audio = whisper.load_audio(audio_file)
audio = whisper.pad_or_trim(audio)
audio = np.array(audio)

# Add random noise
noise = np.random.normal(0, 0.001, (len(audio)))
audio = np.array(audio) + noise
audio = audio.astype(np.float32)

audio_sample_rate = 16000 # 16kHz is what whisper resamples all audio to
write("noised.mp3", rate = audio_sample_rate, data = audio.astype(np.float32))

# Plot the log-mel spectogram
mel = whisper.log_mel_spectrogram(audio).to(model.device)
mel_cpu = mel.cpu().data.numpy()

fig = plt.figure(figsize=(8,2), dpi=100)

plt.yscale("linear")
librosa.display.specshow(mel_cpu, y_axis="log", fmax=16_0000, x_axis="s")

plt.title("Log-mel Spectogram")
plt.colorbar(format='%+2.0f dB');
plt.savefig(graphpng)
print(f'Written log-mel spectrogram to {graphpng}')

_, probs = model.detect_language(mel)
print(f'Detected Language: {max(probs, key=probs.get)}')

opts = whisper.DecodingOptions()
result = whisper.decode(model, mel, opts)

print("Audio features:")
print(result.audio_features.shape)

print('Tokens:')
print(result.tokens)
print('\nText:')
print(result.text)