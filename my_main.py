import pydub
import numpy as np
import scipy.interpolate as sint
from speech_model import ModelSpeech
from speech_model_zoo import SpeechModel251
from speech_features import Spectrogram

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

def load_wav(fn,glb_fs=16000):
    _,suf = fn.rsplit('.',1)
    suf = suf.lower()
    func = getattr(pydub.AudioSegment,f'from_{suf}')
    d = func(fn)
    fs = d.frame_rate
    wave = d.get_array_of_samples()
    wave = np.array(wave)
    if d.channels > 1:
        wave = wave.reshape((-1,d.channels))
        wave = np.mean(wave,axis=1)
    if fs != glb_fs:
        n1 = len(wave)
        n2 = round(n1 * glb_fs / fs)
        x1 = np.linspace(0,1,n1)
        x2 = np.linspace(0,1,n2)
        wave = sint.interp1d(x1,wave,kind='cubic')(x2)
    wave = wave.astype(np.short,copy=False)
    return wave[None,:], glb_fs

def main():
    fn = r"cafe.wav"
    wave,fs = load_wav(fn)
    wave = wave[:,:256000]
    print(wave.shape,fs)
    r = ms.recognize_speech(wave, fs)
    print(r)

if __name__ == '__main__':
    main()

