import matplotlib.pyplot as plt
import numpy as np
import wave
import contextlib
from numpy import fft


# function from http://stackoverflow.com/questions/2226853/interpreting-wav-data/2227174#2227174
def interpret_wav(raw_bytes, n_frames, n_channels, sample_width, interleaved = True):

    if sample_width == 1:
        dtype = np.uint8 # unsigned char
    elif sample_width == 2:
        dtype = np.int16 # signed 2-byte short
    else:
        raise ValueError("Only supports 8 and 16 bit audio formats.")

    channels = np.frombuffer(raw_bytes, dtype=dtype)

    if interleaved:
        # channels are interleaved, i.e. sample N of channel M follows sample N of channel M-1 in raw data
        channels.shape = (n_frames, n_channels)
        channels = channels.T
    else:
        # channels are not interleaved. All samples from channel M occur before all samples from channel M-1
        channels.shape = (n_channels, n_frames)

    return channels


def high_pass_filter(signal, cutoff, fs):
    sig_fft = fft.fft(signal)
    freq = fft.fftfreq(len(signal), d=1/fs)
    sig_fft[np.abs(freq) < cutoff] = 0.001
    filtered = np.real(fft.ifft(sig_fft))
    return filtered


def low_pass_filter(signal, cutoff, fs):
    sig_fft = fft.fft(signal)
    freq = fft.fftfreq(len(signal), d=1/fs)
    sig_fft[np.abs(freq) > cutoff] = 0.001
    filtered = np.real(fft.ifft(sig_fft))
    return filtered


def gain(signal, G):
    gain = 10 ** (G / 20)
    filtered = signal * gain
    return filtered


def speed(sample_rate, rate):
    v = sample_rate * rate
    return v


def gain_in_time(signal, n):
    if n == 0:
        filtered = signal
    else:
        t = np.linspace(0, n * 2 * np.pi, num=len(signal))
        x = abs((np.cos(t) + 1.5) / 2.5)
        filtered = signal * x
    return filtered


def reverse(signal):
    filtered = np.flip(signal)
    return filtered


def menu():
    print("[1] Wczytaj plik .wav ")
    print("[2] Wzmocnij/zmniejsz amplitudę ")
    print("[3] Filtr gornoprzepustowy")
    print("[4] Filtr dolnoprzepustowy")
    print("[5] Przyspieszenie")
    print("[6] Tłumienie w czasie")
    print("[7] Odwróć")
    print("[8] Zapisz")
    print("[0] Wyjscie")


loop = True
while loop:
    menu()
    selection = input("Wybierz opcje : ")

    if selection == "1":
        print("[1] Wczytaj plik .wav ")
        open_error = True
        while open_error:
            try:
                fname = input("Podaj nazwe pliku do wczytania :")
                with contextlib.closing(wave.open(fname, 'rb')) as spf:
                    sampleRate = spf.getframerate()
                    sampWidth = spf.getsampwidth()
                    nChannels = spf.getnchannels()
                    nFrames = spf.getnframes()

                    # Extract Raw Audio from multi-channel Wav File
                    signal = spf.readframes(nFrames * nChannels)
                    spf.close()
                    channels = interpret_wav(signal, nFrames, nChannels, sampWidth, True)
            except Exception as e:
                print(e)
            else:
                open_error = False
                print("##########   Wczytano plik ", fname)
                pass
    elif selection == "2":
        print("[2] Wzmocnij/zmniejsz amplitudę ")
        G = int(input("Podaj wartość wzmocnienia [w dB [-10, 10]]: "))
        while G > 10 or G < -10:
            print("Zła wartość wzmocnienia. Wybierz wzmocnienie z podanego przedziału")
            G = int(input("Podaj wartość wzmocnienia [w dB [-10, 10]]: "))
        filtered = gain(channels[0], G).astype(channels.dtype)
    elif selection == "3":
        print("[3] Filtr gornoprzepustowy")
        cutoff = int(input("Podaj czestotliwosc odciecia [20, 16000]: "))
        while cutoff < 20 or cutoff > 16000:
            print("Zła wartość częstotliwości. Wybierz częstotliwość z podanego przedziału")
            cutoff = int(input("Podaj czestotliwosc odciecia [20, 20000]: "))
        filtered = high_pass_filter(channels[0], cutoff, sampleRate).astype(channels.dtype)
    elif selection == "4":
        print("[4] Filtr dolnoprzepustowy")
        cutoff = int(input("Podaj czestotliwosc odciecia [20, 16000]: "))
        while cutoff < 20 or cutoff > 16000:
            print("Zła wartość częstotliwości. Wybierz częstotliwość z podanego przedziału")
            cutoff = int(input("Podaj czestotliwosc odciecia [20, 20000]: "))
        filtered = low_pass_filter(channels[0], cutoff, sampleRate).astype(channels.dtype)
    elif selection == "5":
        print("[5] Przyspieszenie")
        v = float(input("Podaj przyspieszenie [0.2, 5]: "))
        while v < 0.2 or v > 5:
            print("Zła wartość przyspieszenia. Wybierz wartość z podanego przedziału")
            v = float(input("Podaj przyspieszenie [0.2, 5]: "))
        filtered = channels[0].astype(channels.dtype)
        sampleRate = speed(sampleRate, v)
    elif selection =="6":
        print("[6] Tłumienie w czasie")
        n = int(input("Podaj liczbę tłumień [n > 0]: "))
        while n < 0:
            print("Zła wartość. Podaj wartość z podanego przedziału")
            n = int(input("Podaj liczbę tłumień [n > 0]: "))
        filtered = gain_in_time(channels[0], n).astype(channels.dtype)
    elif selection =="7":
        print("[7] Odwróć")
        filtered = reverse(channels[0]).astype(channels.dtype)
    elif selection == "8":
        print("[8] Zapisz")
        outname = input("Podaj nazwe do zapisu: ")
        wav_file = wave.open(outname, "w")
        wav_file.setparams((1, sampWidth, sampleRate, nFrames, spf.getcomptype(), spf.getcompname()))
        wav_file.writeframes(filtered.tobytes('C'))
        wav_file.close()
    elif selection == "0":
        loop = False
    else:
        print("Bledny wybor")

