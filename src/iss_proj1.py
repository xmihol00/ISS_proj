
#=========================================================================================================
# Soubor:      iss_proj1.py
# Projekt:     VUT, FIT, ISS, zkoumani reci
# Datum:       2. 1. 2021
# Autor:       David Mihola
# Kontakt:     xmihol00@stud.fit.vutbr.cz
# Popis:       Telo projektu, vykreslovani grafu, manipulace s nahravkami
#=========================================================================================================

from scipy.io import wavfile
from scipy import signal
from matplotlib import pyplot as plt
import numpy as np
import functions as fc

save_fig = True    # Ulozeni grafu
save_rec = False    # Ulozeni nahravek
show_plot = False   # Zobrazeni grafu

# Idealne spoustet nezavisle
ukol10 = False      # Vypocet ukolu 10
ukol11 = False      # Vypocet ukolu 11
ukol12 = False      # Vypocet ukolu 12
ukol13 = False      # Vypocet ukolu 13
ukol15 = False      # Vypocet ukolu 15
   
rate_off, moff = wavfile.read('../audio/maskoff_tone.wav')
rate_on, mon = wavfile.read('../audio/maskon_tone.wav')

print("Tón - délky bez roušky\n\t ve vzorcích:", len(moff), "\n\t v sekundách:", len(moff)/rate_off)
print("Tón - délky s rouškou\n\t ve vzorcích:", len(mon), "\n\t v sekundách:", len(mon)/rate_on)

frames_count = 100
frame_size = int(rate_off*0.02)
if ukol15:
    frame_size = int(rate_off*0.025)

# 1.01 s ze stredu nahravky bez masky (pripadne 1.015 s kvuli ukolu 15)
moff_sample = moff[8000-frame_size//2:24000+frame_size//2]

# cross-korelace vzorku nahravky bez masky s nahravkou s maskou
corr = signal.correlate(mon/1024, moff_sample/1024, mode='same')
# vyber 1.01 s (1.015 s) z nahravky s maskou podle nejvetsi korelace
match = mon[np.argmax(abs(corr))-len(moff_sample)//2: np.argmax(abs(corr))+len(moff_sample)//2]

# ustredneni a normalizace bez masky
moff_sample = moff_sample - np.mean(moff_sample)
moff_sample /= np.abs(moff_sample).max()

# ustredneni a normalizace s maskou
mon_sample = match - np.mean(match)
mon_sample /= np.abs(mon_sample).max()

# vytvoreni ramcu
Frames_off, Frames_on = fc.create_frames(moff_sample, mon_sample, frame_size)

# klipovani ramcu
Clipped_off, Clipped_on = fc.clip_frames(Frames_off, Frames_on, frame_size, 0.8 if ukol12 else 0.7)

# autokorelace ramcu
R_off, Freq_off, R_on, Freq_on = fc.autocorelate(Clipped_off, Clipped_on)

# vyhledani nepodobnejsiho ramce
Bframe = fc.best_frame_match(Frames_off, Frames_on)

# oprava dvojnasobneho lagu v ukolu 12
if ukol12:
    fig, axes = plt.subplots(3, 1, constrained_layout=True, figsize=(15,15))

    axes.flat[0].set_title("Základní frekvence rámců před opravou")
    axes.flat[0].plot(Freq_off, label='bez roušky')
    axes.flat[0].plot(Freq_on, label='s rouškou')
    axes.flat[0].legend(loc='upper right')
    axes.flat[0].set_xticks(np.arange(0, 110, 10))
    axes.flat[0].set_xticklabels([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    axes.flat[0].set_xlabel("Rámce")
    axes.flat[0].set_ylabel("$f_0$ $[Hz]$")

    Freq_off, Freq_on, wrong_frame, off_on = fc.correct_frames(Freq_off, Freq_on, frames_count, 10)

    if wrong_frame >= 0:
        wrong = R_on[wrong_frame] if off_on else R_off[wrong_frame]
        axes.flat[1].set_title("Autokorelační koeficienty chybného rámce č. " + str(wrong_frame))
        axes.flat[1].axvline(x=32, color='black', label='Práh')
        lag = [np.argmax(wrong[32:320]) + 32]
        lag_pos = [max(wrong[32:320])]
        axes.flat[1].stem(lag, lag_pos, markerfmt='C3o', linefmt='red', label='Lag')
        axes.flat[1].legend(loc='upper right')
        axes.flat[1].plot(wrong)
        axes.flat[1].set_xticks(np.arange(0, 340, 20))
        axes.flat[1].set_xticklabels([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320])
        axes.flat[1].set_xlabel("Vzorky")
        axes.flat[1].set_ylabel("velikost korelace")

    axes.flat[2].set_title("Základní frekvence rámců po opravě")
    axes.flat[2].plot(Freq_off, label='bez roušky')
    axes.flat[2].plot(Freq_on, label='s rouškou')
    axes.flat[2].legend(loc='upper right')
    axes.flat[2].set_ylim([int(np.mean(Freq_off)) - 3 if np.mean(Freq_off) > np.mean(Freq_on) else int(np.mean(Freq_on)) - 3, 
                           int(np.mean(Freq_off)) + 3 if np.mean(Freq_off) < np.mean(Freq_on) else int(np.mean(Freq_on)) + 3])
    axes.flat[2].set_xticks(np.arange(0, 110, 10))
    axes.flat[2].set_xticklabels([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    axes.flat[2].set_xlabel("Rámce")
    axes.flat[2].set_ylabel("$f_0$ $[Hz]$")
    if save_fig:
        plt.savefig('../ukol12.png')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

# fazovy posun ramcu v ukolu 15
if ukol15:
    fig, axes = plt.subplots(3, 1, constrained_layout=True, figsize=(15,15))
    axes.flat[0].set_title("Rámec č. 81 z úseku s rouškou a bez")
    axes.flat[0].plot(Frames_off[81][0:320], label='bez roušky')
    axes.flat[0].plot(Frames_on[81][0:320], label='s rouškou')
    axes.flat[0].legend(loc='upper right')
    axes.flat[0].set_xticks(np.arange(0, 340, 20))
    axes.flat[0].set_xticklabels([0, 1.25, 2.5, 3.75, 5, 6.25, 7.5, 8.75, 10, 11.25, 13.75, 12.5, 15, 16.25, 17.5, 18.75, 20])
    axes.flat[0].set_xlabel("Čas $[ms]$")
    axes.flat[0].set_ylabel("Ustředněné normalizované hodnoty")
    Frames_off, Clipped_off, Frames_on, Clipped_on, phase = fc.phase_shift(Frames_off, Clipped_off, Frames_on, Clipped_on, 155)

    axes.flat[1].set_title("Rámec č. 81 z úseku s rouškou a bez")
    axes.flat[1].plot(Frames_off[81], label='bez roušky')
    axes.flat[1].plot(Frames_on[81], label='s rouškou')
    axes.flat[1].legend(loc='upper right')
    axes.flat[1].set_xticks(np.arange(0, 340, 20))
    axes.flat[1].set_xticklabels([0, 1.25, 2.5, 3.75, 5, 6.25, 7.5, 8.75, 10, 11.25, 13.75, 12.5, 15, 16.25, 17.5, 18.75, 20])
    axes.flat[1].set_xlabel("Čas $[ms]$")
    axes.flat[1].set_ylabel("Ustředněné normalizované hodnoty")

    axes.flat[2].set_title("Relativní fázový posun rámce s rouškou k rámci bez roušky")
    axes.flat[2].plot(phase)
    axes.flat[2].set_xlabel("Rámec")
    axes.flat[2].set_ylabel("Posunu ve vzorcích")
    if save_fig:
        plt.savefig('../ukol15.png')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

fig = plt.figure(figsize=(15, 6))
plt.title("Rámec č. " + str(Bframe) + " z úseku s rouškou a bez")
plt.plot(Frames_off[Bframe], label='bez roušky')
plt.plot(Frames_on[Bframe], label='s rouškou')
plt.legend(loc='upper right')
plt.xticks(np.arange(0, 340, 20), [0, 1.25, 2.5, 3.75, 5, 6.25, 7.5, 8.75, 10, 11.25, 13.75, 12.5, 15, 16.25, 17.5, 18.75, 20])
plt.xlabel("Čas $[ms]$")
plt.ylabel("Ustředněné normalizované hodnoty")
if save_fig:
    plt.savefig('../ukol3.png')
if show_plot:
    plt.show()
else:
    plt.close(fig)

fig, axes = plt.subplots(1, 1, constrained_layout=True, figsize=(15,5))
axes.plot(Frames_off[Bframe])
axes.set_title("Rámec č. " + str(Bframe))
axes.set_xticks(np.arange(0, 340, 20))
axes.set_xticklabels([0, 1.25, 2.5, 3.75, 5, 6.25, 7.5, 8.75, 10, 11.25, 13.75, 12.5, 15, 16.25, 17.5, 18.75, 20])
axes.set_xlabel("Čas $[ms]$")
axes.set_ylabel("Ustředněné norm. hodnoty")
if save_fig:
    plt.savefig('../ukol4a.png')
if show_plot:
    plt.show()
else:
    plt.close(fig)

fig, axes = plt.subplots(3, 1, constrained_layout=True, figsize=(15,15))
axes.flat[0].set_title("Centrální klipování rámece č. " + str(Bframe) + " s 70%")
axes.flat[0].set_xticks(np.arange(0, 340, 20))
axes.flat[0].set_xticklabels([0, 1.25, 2.5, 3.75, 5, 6.25, 7.5, 8.75, 10, 11.25, 13.75, 12.5, 15, 16.25, 17.5, 18.75, 20])
axes.flat[0].set_xlabel("Čas $[ms]$")
axes.flat[0].set_ylabel("Hodnoty po klipování")
axes.flat[0].plot(Clipped_off[Bframe])

axes.flat[1].set_title("Autokorelace rámece č. " + str(Bframe))
axes.flat[1].axvline(x=32, color='black', label='Práh')
lag = [np.argmax(R_off[Bframe][32:320]) + 32]
lag_pos = [max(R_off[Bframe][32:320])]
axes.flat[1].stem(lag, lag_pos, markerfmt='C3o', linefmt='red', label='Lag')
axes.flat[1].legend(loc='upper right')
axes.flat[1].plot(R_off[Bframe])
axes.flat[1].set_xticks(np.arange(0, 340, 20))
axes.flat[1].set_xticklabels([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320])
axes.flat[1].set_xlabel("Vzorky")
axes.flat[1].set_ylabel("velikost korelace")

print("Bez roušky\n\tstřední hodnota: ", np.mean(Freq_off), "\n\trozptyl: ", np.var(Freq_off))
print("S rouškou\n\tstřední hodnota: ", np.mean(Freq_on), "\n\trozptyl: ", np.var(Freq_on))

axes.flat[2].set_title("Základní frekvence rámců")
axes.flat[2].plot(Freq_off, label='bez roušky')
axes.flat[2].plot(Freq_on, label='s rouškou')
axes.flat[2].set_ylim([int(np.mean(Freq_off)) - 3 if np.mean(Freq_off) > np.mean(Freq_on) else int(np.mean(Freq_on)) - 3, 
                       int(np.mean(Freq_off)) + 3 if np.mean(Freq_off) < np.mean(Freq_on) else int(np.mean(Freq_on)) + 3])
axes.flat[2].legend(loc='upper right')
axes.flat[2].set_xticks(np.arange(0, 110, 10))
axes.flat[2].set_xticklabels([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
axes.flat[2].set_xlabel("Rámce")
axes.flat[2].set_ylabel("$f_0$ $[Hz]$")
if save_fig:
    plt.savefig('../ukol4b.png')
if show_plot:
    plt.show()
else:
    plt.close(fig)

N = 1024

# aplikace okenkove funkce
if ukol11:
    # okenkova funkce
    hannning_w = np.hanning(320)

    # impulsni odezva okenkove funkce
    HW_response = 10*np.log10(np.abs(np.fft.fft(hannning_w, 1024)+1e-20))

    fig, axes = plt.subplots(3, 1, constrained_layout=True, figsize=(15,17))
    axes.flat[0].set_title("Hanningova okénková funkce")
    axes.flat[0].plot(hannning_w)
    axes.flat[0].set_xlabel("Vzorky")
    axes.flat[0].set_ylabel("Hodnoty vzorků")
    axes.flat[1].set_title("Impulsní odezva Hanningovy funkce")
    axes.flat[1].plot(HW_response[0:512])
    axes.flat[1].set_xlabel("Frekvence $[Hz]$")
    axes.flat[1].set_xticks(np.arange(0, 576, 64))
    axes.flat[1].set_xticklabels([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000])
    axes.flat[1].set_ylabel("Spektralní hustota výkonu $[dB]$")

    frame_no_win = 10*np.log10(np.abs(np.fft.fft(Frames_off[Bframe], n=1024)+1e-20))
    Frames_off, Frames_on = fc.apply_window(hannning_w, Frames_off, Frames_on)
    frame_win = 10*np.log10(np.abs(np.fft.fft(Frames_off[Bframe], n=1024)+1e-20))

    axes.flat[2].set_title("Spektrum rámce č. " + str(Bframe) + " bez okenkové funkce")
    axes.flat[2].plot(frame_no_win[0:512], label='před aplikací okénkové funkce')
    axes.flat[2].plot(frame_win[0:512], label='po aplikaci okénkové funkce')
    axes.flat[2].legend(loc='upper right')
    axes.flat[2].set_xticks(np.arange(0, 576, 64))
    axes.flat[2].set_xticklabels([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000])
    axes.flat[2].set_xlabel("Frekvence $[Hz]$")
    axes.flat[2].set_ylabel("Spektralní hustota výkonu $[dB]$")
    if save_fig:
        plt.savefig('../ukol11.png')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

# vyber ramcu se stejnou zakladni frekvenci pro kazdy ramec
label13 = ""
if ukol13:
    Frames_off, Frames_on, frames_count = fc.same_freqv_frames(Frames_off, Frames_on, Freq_off, Freq_on)
    print("počet stejných rámců:", frames_count)
    label13 = " s rámci se stejnou základní frekvencí"

## moje DFT - pomala :(
#DFT_off, DFT_on = fc.calculate_DFT(Frames_off, Frames_on, frames_count)
## radsi pouzit FFT, vysledek stejny
DFT_off, DFT_on = fc.use_FFT(Frames_off, Frames_on, frames_count)

# prevod na decibely
Spect_off = fc.decibel_tranform(DFT_off, frames_count)
Spect_on = fc.decibel_tranform(DFT_on, frames_count)

fig = plt.figure(figsize=(12, 10))
plt.title("Spektrogram bez roušky")
plt.pcolormesh([list(i) for i in zip(*[Spect_off[j] for j in range(0, frames_count)])])
plt.gca().set_xlabel('Čas $[s]$')
plt.gca().set_ylabel('Frekvence $[Hz]$')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu $[dB]$', rotation=270, labelpad=15)
plt.yticks([0, 9.92, 19.84, 29.76, 39.68, 49.6, 59.52, 69.44, 79.36, 89.28, 99.2, 128, 192, 256, 320, 384, 448, 512], 
           [0, 155, 310, 465, 620, 775, 930, 1085, 1240, 1395, 1550, 2000, 3000, 4000, 5000, 6000, 7000, 8000])
plt.xticks(np.arange(0, 110, 10), [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
if save_fig:
    plt.savefig('../nomask_spec.png')
if show_plot:
    plt.show()
else:
    plt.close(fig)

fig = plt.figure(figsize=(12, 10))
plt.title("Spektrogram s rouškou")
plt.pcolormesh([list(i) for i in zip(*[Spect_on[j] for j in range(0, frames_count)])])
plt.gca().set_xlabel('Čas $[s]$')
plt.gca().set_ylabel('Frekvence $[Hz]$')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu $[dB]$', rotation=270, labelpad=15)
plt.yticks([0, 9.92, 19.84, 29.76, 39.68, 49.6, 59.52, 69.44, 79.36, 89.28, 99.2, 128, 192, 256, 320, 384, 448, 512], 
           [0, 155, 310, 465, 620, 775, 930, 1085, 1240, 1395, 1550, 2000, 3000, 4000, 5000, 6000, 7000, 8000])
plt.xticks(np.arange(0, 110, 10), [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
if save_fig:
    plt.savefig('../mask_spec.png')
if show_plot:
    plt.show()
else:
    plt.close(fig)

# vypocet frekvencni charakteristiky
Freq_char = fc.freqv_char(DFT_off, DFT_on, frames_count)

# prumer pres vsechny ramce
Freq_avrg = fc.freqv_avg(Freq_char, frames_count)

Freq_spect = []
for i in range(0, 512):
    Freq_spect.append(10*np.log10((Freq_avrg[i]+1e-20)**2))

fig = plt.figure(figsize=(15, 6))
plt.title("Frekvenční charakteristika roušky" + label13)
plt.plot(Freq_spect)
plt.xticks([0, 40, 64, 100, 128, 192, 256, 320, 384, 448, 512], [0, 625, 1000, 1562.5, 2000, 3000, 4000, 5000, 6000, 7000, 8000])
plt.xlabel("Frekvence $[Hz]$")
plt.ylabel("Spektralní hustota výkonu $[dB]$")
if save_fig:
    if ukol13:
        plt.savefig('../ukol13.png')
    else:    
        plt.savefig('../ukol6.png')
if show_plot:
    plt.show()
else:
    plt.close(fig)

# vypocet IDFT
Filter = fc.calculate_IDFT(Freq_avrg)

Filter_real = []
for i in range(0, N//2):
    Filter_real.append(Filter[i].real)

fig = plt.figure(figsize=(15, 6))
plt.title("Impulsní odezva roušky")
plt.plot(Filter_real)
plt.xlabel("Vzorky")
plt.ylabel("Hodnota impulsní odezvy")
if save_fig:
    plt.savefig('../impuls_resp.png')
if show_plot:
    plt.show()
else:
    plt.close(fig)

rate_soff, smoff = wavfile.read('../audio/maskoff_sentence.wav')
rate_son, smon = wavfile.read('../audio/maskon_sentence.wav')

print("Věta - délky bez roušky\n\t ve vzorcích:", len(smoff), "\n\t v sekundách:", len(smoff)/rate_soff)
print("Věta - délky s rouškou\n\t ve vzorcích:", len(smon), "\n\t v sekundách:", len(smon)/rate_son)

sfilter = None
tfilter = None
if ukol10:
    # aplikace filtru metodou overlap-add
    sfilter = fc.overlap_add(Filter_real, smoff, N)
    tfilter = fc.overlap_add(Filter_real, moff, N)
else:
    sfilter = signal.lfilter(Filter_real, [1.0], smoff)
    tfilter = signal.lfilter(Filter_real, [1.0], moff)

fig, axes = plt.subplots(4, 1, constrained_layout=True, figsize=(15,20))
axes.flat[0].set_title("Nahrávky vět")
axes.flat[0].plot(smon, label='s rouškou')
axes.flat[0].plot(smoff, label='bez roušky')
axes.flat[0].legend(loc='upper right')
axes.flat[0].set_xlabel("Vzorky")
axes.flat[0].set_ylabel("Hodnota signálu")

axes.flat[1].set_title("Nahrávky vět")
axes.flat[1].plot(smon, label='s rouškou')
axes.flat[1].plot(sfilter, label='se simulovanou rouškou')
axes.flat[1].legend(loc='upper right')
axes.flat[1].set_xlabel("Vzorky")
axes.flat[1].set_ylabel("Hodnota signálu")

axes.flat[2].set_title("Nahrávky vět")
axes.flat[2].plot(sfilter, label='se simulovanou rouškou')
axes.flat[2].plot(smoff, label='bez roušky')
axes.flat[2].legend(loc='upper right')
axes.flat[2].set_xlabel("Vzorky")
axes.flat[2].set_ylabel("Hodnota signálu")

axes.flat[3].set_title("Nahrávky vět přiblíženě")
axes.flat[3].plot(sfilter[10:], label='se simulovanou rouškou')
axes.flat[3].plot(smoff[10:], label='bez roušky')
axes.flat[3].plot(smon, label='s rouškou')
axes.flat[3].legend(loc='upper right')
axes.flat[3].set_xlabel("Vzorky")
axes.flat[3].set_ylabel("Hodnota signálu")
axes.flat[3].set_xlim([63500, 64500])
if save_fig:
    plt.savefig('../ukol8.png')
if show_plot:
    plt.show()
else:
    plt.close(fig)


file_name_s = ""
file_name_t = ""

if not ukol10 and not ukol11 and not ukol12 and not ukol13 and not ukol15:
    file_name_s = "../audio/sim_maskon_sentence.wav"
    file_name_t = "../audio/sim_maskon_tone.wav"
elif ukol10 and not ukol11 and not ukol12 and not ukol13 and not ukol15:
    file_name_s = "../audio/sim_maskon_sentence_overlap_add.wav"
    file_name_t = "../audio/sim_maskon_tone_overlap_add.wav"
elif not ukol10 and ukol11 and not ukol12 and not ukol13 and not ukol15:
    file_name_s = "../audio/sim_maskon_sentence_window.wav"
    file_name_t = "../audio/sim_maskon_tone_window.wav"
elif not ukol10 and not ukol11 and not ukol12 and ukol13 and not ukol15:
    file_name_s = "../audio/sim_maskon_sentence_only_match.wav"
    file_name_t = "../audio/sim_maskon_tone_only_match.wav"
elif not ukol10 and not ukol11 and not ukol12 and not ukol13 and ukol15:
    file_name_s = "../audio/sim_maskon_sentence_phase.wav"
    file_name_t = "../audio/sim_maskon_tone_phase.wav"

# ulozeni nahravek
if len(file_name_s) > 1 and save_rec:
    wavfile.write(file_name_s, 16000, np.int16(sfilter))
    wavfile.write(file_name_t, 16000, np.int16(tfilter))

