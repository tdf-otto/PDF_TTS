# This is a revamped version of my original code which takes a .txt file as input and outputs a .wav file.


from kokoro import KPipeline # install 
from IPython.display import display, Audio
import soundfile as sf
import torch
import numpy as np
import os
import tkinter

pipeline = KPipeline(lang_code='a')
# Load the voice. May be af_sarah, am_liam, or any other voice installed from
# https://huggingface.co/voices
voice_tensor = torch.load('./voices/am_liam.pt', weights_only=True) 

# Create texts and soundfiles directories if they don't exist
if not os.path.exists('texts'):
    os.makedirs('texts', exist_ok=True)
if not os.path.exists('soundfiles'):
    os.makedirs('soundfiles', exist_ok=True)

# have user select a text file from the texts directory with a simple command line input
document = input("Enter the name of the text file (including extension; must be in /texts folder): ")

with open("texts/"+document, 'r', encoding='utf-8') as f:
    print("filetype " + document[document.find("."):])
    text = f.read()

generator = pipeline(text, voice=voice_tensor, speed=1.0, split_pattern=r'\n+')

sr = 24000
silence = np.zeros(int(sr * 0.5), dtype=np.float32)
chunks = []

for i, (gs, ps, audio) in enumerate(generator):
    print(i, gs, ps)
    display(Audio(data=audio, rate=sr, autoplay=i == 0))
    chunks.append(audio.detach().cpu().numpy().astype(np.float32))
    chunks.append(silence)

if chunks:
    final_audio = np.concatenate(chunks)
    sf.write("soundfiles/" + document[:document.find(".")] + ".wav", final_audio, sr)