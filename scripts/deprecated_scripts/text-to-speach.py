
import torch
import librosa
#from inference import Mars5TTS, InferenceConfig as config_class


# mars5, config_class = torch.hub.load('Camb-ai/mars5-tts', 'mars5_english', trust_repo=True)
# # The `mars5` contains the AR and NAR model, as well as inference code.
# # The `config_class` contains tunable inference config settings like temperature.

# # Load reference audio between 1-12 seconds.
# wav, sr = librosa.load('voice_1.wav',
#                        sr=mars5.sr, mono=True)


# wav = torch.from_numpy(wav)
# ref_transcript = " Every morning I start my day with a cup of coffee and a short walk \
# , it helps me feel refreshed and ready for whatever comes next."

# # Pick whether you want a deep or shallow clone. Set to False if you don't know prompt transcript or want fast inference. Set to True if you know transcript and want highest quality.
# deep_clone = True
# # Below you can tune other inference settings, like top_k, temperature, top_p, etc...
# cfg = config_class(deep_clone=deep_clone, rep_penalty_window=100,
#                       top_k=100, temperature=0.7, freq_penalty=3)

# ar_codes, output_audio = mars5.tts("Hello, my name is John and i live there.", wav,
#           ref_transcript,
#           cfg=cfg)
# # output_audio is (T,) shape float tensor corresponding to the 24kHz output audio.

# #save the audio in wav format
# import torchaudio
# torchaudio.save("output_audio.wav", output_audio.unsqueeze(0), 24000)

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

config = XttsConfig()
config.load_json("/path/to/xtts/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="/path/to/xtts/", eval=True)
model.cuda()

outputs = model.synthesize(
    "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
    config,
    speaker_wav="/data/TTS-public/_refclips/3.wav",
    gpt_cond_len=3,
    language="en",
)
