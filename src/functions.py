
#=========================================================================================================
# Soubor:      functions.py
# Projekt:     VUT, FIT, ISS, zkoumani reci
# Datum:       2. 1. 2021
# Autor:       David Mihola
# Kontakt:     xmihol00@stud.fit.vutbr.cz
# Popis:       Funkce implementujici hlavni funckionalitu projektu
#=========================================================================================================

from scipy.io import wavfile
from scipy import signal
from matplotlib import pyplot as plt
import numpy as np

def create_frames(moff_sample, mon_sample, size):
    frames_off = []
    frames_on = []
    for i in range(0, 100):
        frames_off.append(moff_sample[160*i:160*i+size])
        frames_on.append(mon_sample[160*i:160*i+size])
    
    return frames_off, frames_on

def phase_shift(frames_off, clipped_off, frames_on, clipped_on, f0):
    phase = []
    for i in range(0, 100):
        # korelace
        corr_off_on = np.correlate(clipped_off[i], clipped_on[i], mode='same')
        corr_on_off = np.correlate(clipped_on[i], clipped_off[i], mode='same')
        # velikost posunu

        max_off_on = abs(np.argmax(corr_off_on)-199)
        max_on_off = abs(np.argmax(corr_on_off)-199)

        if max_off_on > 80 and max_on_off > 80:
            # korekce dvojiteho posunu
            max_off_on = abs(max_off_on - 16000//f0)
            max_on_off = abs(max_on_off - 16000//f0)

        if max_off_on < max_on_off:
            # posun ramcu
            phase.append(max_off_on) # zaznamenani velikosti posunu
            frames_on[i] = frames_on[i][max_off_on:max_off_on+320]
            frames_off[i] = frames_off[i][0:320]
            clipped_on[i] = clipped_on[i][max_off_on:max_off_on+320]
            clipped_off[i] = clipped_off[i][0:320]
        else:
            phase.append(-max_on_off) # zaznamenani velikosti posunu
            frames_off[i] = frames_off[i][max_on_off:max_on_off+320]
            frames_on[i] = frames_on[i][0:320]
            clipped_off[i] = clipped_off[i][max_on_off:max_on_off+320]
            clipped_on[i] = clipped_on[i][0:320]

    return frames_off, clipped_off, frames_on, clipped_on, phase

def best_frame_match(frames_off, frames_on):
    corr_res = []
    for i in range(0, 100):
        corr_res.append(max(signal.correlate(frames_off[i], frames_on[i], mode='same')))
    
    return np.argmax(corr_res)

def clip_frames(frames_off, frames_on, size, clip):
    clipped_on = []
    clipped_off = []
    for i in range(0, 100):
        on = []
        off = []
        # vypocet klipovacich hodnot
        on_treshold = max(abs(frames_on[i]))*clip
        off_treshold = max(abs(frames_off[i]))*clip
        for j in range(0, size):
            # aplikace klipovaci urovne na kazdy vzorek v ramci
            if on_treshold < frames_on[i][j]:
                on.append(1)
            elif on_treshold < -frames_on[i][j]:
                on.append(-1)
            else:
                on.append(0)

            if off_treshold < frames_off[i][j]:
                off.append(1)
            elif off_treshold < -frames_off[i][j]:
                off.append(-1)
            else:
                off.append(0)  
        clipped_on.append(on)
        clipped_off.append(off)

    return clipped_off, clipped_on

def autocorelate(clipped_off, clipped_on):
    freq_off = []
    freq_on = []
    R_off = []
    R_on = []
    for i in range(0, 100):
        rn_off = []
        rn_on = []
        for k in range(0, 320):
            sum_off = 0
            sum_on = 0
            for n in range(0, 320 - k):
                # aplikace vzorce
                sum_off += clipped_off[i][n]*clipped_off[i][n+k]
                sum_on += clipped_on[i][n]*clipped_on[i][n+k]
            rn_off.append(sum_off)
            rn_on.append(sum_on)
        R_off.append(rn_off)
        R_on.append(rn_on)
        rn_off = rn_off[32:320] # 16000/500 = 32 --> prah
        rn_on = rn_on[32:320]
        freq_off.append(16000/(rn_off.index(max(rn_off)) + 32)) # + 32 -> pricteni prahu
        freq_on.append(16000/(rn_on.index(max(rn_on)) + 32))
    
    return R_off, freq_off, R_on, freq_on

def calculate_DFT(frames_off, frames_on, frames_count):
    DFT_off = []
    DFT_on = []
    for i in range(0, frames_count): # pro vsechny ramce
        frame_dft_off = []
        frame_dft_on = []
        for k in range(0, 1024):
            sum_off = 0 + 0j
            sum_on = 0 + 0j
            for n in range(0, 320): # pro n od 320 do 1024 je hodnota vzdy 0
                e_pow = np.e**(-1j*2*np.pi/1024*n*k) # vypocet imaginarni slozky
                # aplikace vzorce
                sum_off += frames_off[i][n]*e_pow
                sum_on += frames_on[i][n]*e_pow
            frame_dft_off.append(sum_off)
            frame_dft_on.append(sum_on)
        DFT_off.append(frame_dft_off)
        DFT_on.append(frame_dft_on)
    return DFT_off, DFT_on

def use_FFT(frames_off, frames_on, frames_count): # poziti FFT namisto me implementace DFT, ktera je pomala
    FFT_off = []
    FFT_on = []
    for i in range(0, frames_count):
        FFT_off.append(np.fft.fft(frames_off[i], n=1024))
        FFT_on.append(np.fft.fft(frames_on[i], n=1024))
    
    return FFT_off, FFT_on

def decibel_tranform(DFT, frames_count):
    spect = []
    for i in range(0, frames_count):
        s = []
        for j in range(0, 512):
            s.append(10*np.log10(abs(DFT[i][j]+1e-20)**2))
        spect.append(s)
    
    return spect

def freqv_char(DFT_off, DFT_on, frames_count):
    freq_char = []
    for i in range(0, frames_count):
        f_char = []
        for j in range(0, 1024):
            f_char.append(DFT_on[i][j]/DFT_off[i][j]) # H = Y(wj)/X(wj)
        freq_char.append(f_char)
    
    return freq_char

def freqv_avg(freqv_char, frames_count):
    Freqv_avrg = []
    for i in range(0, 1024):
        avg = 0
        for j in range(0, frames_count):
            avg += abs(freqv_char[j][i])
        avg /= frames_count
        Freqv_avrg.append(avg)
    
    return Freqv_avrg

def calculate_IDFT(freqv_avg):
    Filter = []
    for n in range(0, 1024):
        s = 0 + 0j
        for k in range(0, 1024):
            # aplikace vzorce
            s += freqv_avg[k]*np.e**(1j*2*np.pi/1024*n*k)
        s /= 1024 # nakonec vydeleni poctem vzorku
        Filter.append(s)
    
    return Filter

def apply_window(window, frames_off, frames_on):
    for i in range(0, 100):
        for j in range(0, 320):
            # aplikace okenkove funkce nasobenim
            frames_off[i][j] *= window[j]
            frames_on[i][j] *= window[j]
    
    return frames_off, frames_on

def overlap_add(flter, signal, fft_n):
    f_len = len(flter)  # delka filteru
    s_len = len(signal) # delka filtrovaneho signalu
    step = fft_n - f_len + 1 # delka jednoho vypocetniho kroku
    filer_FFT = np.fft.fft(flter, n=fft_n)
    res = [0]*(s_len+fft_n)
    for i in range(0, s_len, step):
        segment = np.fft.ifft(np.fft.fft(signal[i:i+step], n=fft_n)*filer_FFT)
        for j in range(0, fft_n): # pricteni na spravnou pozici ve vyfiltrovanem signalu
            res[i+j] += segment[j].real
    
    return res[0:s_len]

def same_freqv_frames(frames_off, frames_on, freqv_off, freqv_on):
    same_off = []
    same_on = []
    match = 0
    for i in range(0, 100):
        if freqv_off[i] == freqv_on[i]:
            # stejna zakladni frekvence
            same_off.append(frames_off[i])
            same_on.append(frames_on[i])
            match += 1
    
    return frames_off, frames_on, match

def correct_frames(freqv_off, freqv_on, frames_count, eps):
    off_median = np.median(freqv_off)
    on_median = np.median(freqv_on)
    wrong_frame = -1 #staci pouze jeden spatny ramec pro graf
    off_on = False
    for i in range(0, frames_count):
        if abs(freqv_off[i] - off_median) > eps:
            # chybny ramec v nahravce bez masky, aplikace medianu
            freqv_off[i] = off_median
            wrong_frame = i
            off_on = False #spatny ramec je ramec bez masky
        if abs(freqv_on[i] - on_median) > eps:
            # chybny ramec v nahravce s maskou, aplikace medianu
            freqv_on[i] = on_median
            wrong_frame = i
            off_on = True #spatny ramec je ramec s maskou
    
    return freqv_off, freqv_on, wrong_frame, off_on

