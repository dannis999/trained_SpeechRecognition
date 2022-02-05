import numpy as np
from speech_model import ModelSpeech
from speech_model_zoo import SpeechModel251
from speech_features import Spectrogram
from utils.ops import read_wav_data

AUDIO_LENGTH = 1600
AUDIO_FEATURE_LENGTH = 200
CHANNELS = 1
OUTPUT_SIZE = 1428
sm251 = SpeechModel251(
    input_shape=(AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, CHANNELS),
    output_size=OUTPUT_SIZE
    )
feat = Spectrogram()
ms = ModelSpeech(sm251, feat, max_label_length=64)
ms.load_model('save_models/' + sm251.get_model_name() + '.model.h5')

def load_wav(fn):
    wave_data, framerate, num_channel, num_sample_width = read_wav_data(fn)
    if num_channel > 1:
        wave_data = np.mean(wave_data,axis=0,keepdims=True,dtype=wave_data.dtype)
    return wave_data, framerate

def main():
    fn = r"D4_752.wav"
    wave,fs = load_wav(fn)
    print(wave.shape,fs)
    r = ms.recognize_speech(wave, fs)
    print(r)

if __name__ == '__main__':
    main()

