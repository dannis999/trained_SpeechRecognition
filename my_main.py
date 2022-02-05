import os,pydub
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
ms.load_model(os.path.join('save_models','SpeechModel251.model.h5'))

def load_wav(fn,glb_fs=16000):
    print('正在读取音频文件')
    _,suf = fn.rsplit('.',1)
    suf = suf.lower()
    func = getattr(pydub.AudioSegment,f'from_{suf}')
    d = func(fn)
    fs = d.frame_rate
    wave = d.get_array_of_samples()
    wave = np.array(wave)
    if d.channels > 1:
        print('正在合并声道')
        wave = wave.reshape((-1,d.channels))
        wave = np.mean(wave,axis=1)
    if fs != glb_fs:
        print('正在重采样')
        n1 = len(wave)
        n2 = round(n1 * glb_fs / fs)
        x1 = np.linspace(0,1,n1)
        x2 = np.linspace(0,1,n2)
        wave = sint.interp1d(x1,wave,kind='cubic')(x2)
    wave = wave.astype(np.short,copy=False)
    return wave, glb_fs

def recog_file(fn_wav:str,fn_pinyin:str,sep_sec=16,overlap_sec=1):
    wave,fs = load_wav(fn_wav)
    print('正在识别')
    n = len(wave)
    sep = round(sep_sec * fs)
    overlap = round(overlap_sec * fs)
    step = sep - overlap
    i = 0
    with open(fn_pinyin,'w',encoding='utf-8') as f:
        while 1:
            dn = min(i+sep,n)
            dwave = wave[i:dn]
            dr = ms.recognize_speech(dwave[None,:], fs)
            sec_start = round(i / fs)
            if dr:
                m,s = divmod(sec_start,60)
                h,m = divmod(m,60)
                ts = '{0:02}:{1:02}:{2:02}'.format(h,m,s)
                ps = ' '.join(dr)
                f.write(f'[{ts}]{ps}\n')
                f.flush()
            if dn >= n:break
            i += step

def main():
    fn1 = input()
    fn2 = input()
    recog_file(fn1,fn2)

if __name__ == '__main__':
    main()
