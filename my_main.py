import os,pydub,gc
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

def traverse_wav(fn,glb_fs=16000,sep_sec=16,overlap_sec=1):
    print('正在读取音频文件')
    _,suf = fn.rsplit('.',1)
    suf = suf.lower()
    func = getattr(pydub.AudioSegment,f'from_{suf}')
    d = func(fn)
    del func
    print('正在转换数据')
    fs = d.frame_rate
    b = d.raw_data
    channels = d.channels
    array_type = d.array_type
    del d
    gc.collect()
    wave = np.frombuffer(b,dtype=array_type)
    del b
    gc.collect()
    if channels > 1:
        print('正在合并声道')
        wave = wave.reshape((-1,channels))
        wave = np.mean(wave,axis=1,dtype=wave.dtype)
    print('正在识别')
    n = len(wave)
    sep = round(sep_sec * fs)
    overlap = round(overlap_sec * fs)
    step = sep - overlap
    if fs != glb_fs:
        n1 = sep
        n2 = round(n1 * glb_fs / fs)
        x1 = np.linspace(0,1,n1)
        x2 = np.linspace(0,1,n2)
    i = 0
    while 1:
        dn = min(i+sep,n)
        dwave = wave[i:dn]
        if fs != glb_fs:
            dx1 = x1[:len(dwave)]
            dx2 = x2[x2 <= dx1[-1]]
            dwave = sint.interp1d(dx1,dwave,kind='cubic')(dx2)
        dwave = dwave.astype(np.short,copy=False)
        sec_start = round(i / fs)
        yield dwave,glb_fs,sec_start
        if dn >= n:break
        i += step

def recog_file(fn_wav:str,fn_pinyin:str,sep_sec=16,overlap_sec=1):
    with open(fn_pinyin,'w',encoding='utf-8') as f:
        for wave,fs,sec_start in traverse_wav(fn_wav):
            dr = ms.recognize_speech(wave[None,:], fs)
            if dr:
                m,s = divmod(sec_start,60)
                h,m = divmod(m,60)
                ts = '{0:02}:{1:02}:{2:02}'.format(h,m,s)
                ps = ' '.join(dr)
                f.write(f'[{ts}]{ps}\n')
                f.flush()

def main():
    fn1 = input()
    fn2 = input()
    recog_file(fn1,fn2)

if __name__ == '__main__':
    main()
