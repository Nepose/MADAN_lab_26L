"""
Metody Analizy Danych – Lab 2
Analiza sygnałów z przetwornika ADC: DFT, DCT, rekonstrukcja

Poprawiona wersja: właściwe ograniczenie dziedziny częstotliwości,
poprawione etykiety DCT, lepsza prezentacja graficzna.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, dct

df = pd.read_csv("data.csv", usecols=[1, 2, 3, 4, 5, 6])
df['Increment'] = df['Increment'].ffill()

T  = df.iloc[0]['Increment']   # okres próbkowania [s]
fs = 1.0 / T                   # częstotliwość próbkowania [Hz]
N  = len(df)                   # liczba próbek

t = np.arange(N) * T

channels = [df['CH1'].values, df['CH2'].values,
            df['CH3'].values, df['CH4'].values]
labels   = ['CH1', 'CH2', 'CH3', 'CH4']

print(f"Parametry sygnału:")
print(f"  N = {N} próbek")
print(f"  T = {T:.2e} s  →  fs = {fs/1e3:.1f} kHz")
print(f"  Czas całkowity = {N*T*1e3:.2f} ms")
print(f"  Rozdzielczość DFT  Δf = {fs/N:.2f} Hz")
print(f"  Częstotliwość Nyquista = {fs/2/1e3:.1f} kHz\n")

freqs    = fftfreq(N, T)       # pełna siatka: –fs/2 … +fs/2
pos_mask = freqs >= 0
freqs_pos = freqs[pos_mask]    # tylko f ≥ 0

def compute_dft(ch):
    """Zwraca (X, amp_pos, phase_pos) dla częstotliwości nieujemnych."""
    X     = fft(ch)
    amp   = np.abs(X) / N
    amp[1:] = 2 * amp[1:]            # podwojenie dla f > 0 (widmo jednostronne)
    phase = np.angle(X)
    # Zerowanie fazy dla składowych o zaniedbywalnej amplitudzie
    threshold = np.max(amp) * 0.01
    phase[amp < threshold] = 0
    return X, amp[pos_mask], phase[pos_mask]

'''
Zakres wyświetlania widma DFT
    Dominujące składowe CH1 i CH2 kończą się ok. 10 kHz.
    CH3 to szum – nie ma dominujących prążków; pokazujemy do 15 kHz.
    CH4 ma składową stałą + harmoniczne do ~5 kHz.
    Wspólny limit: 12 500 Hz (= 30× częstotliwość podstawowa ≈ 833 Hz) odpowiada indeksowi 30 przy Δf ≈ 417 Hz.
'''
F_MAX_DISPLAY = 12_500          # [Hz] – górna granica osi X dla DFT
dft_mask_plot = freqs_pos <= F_MAX_DISPLAY

''' 
Zadanie 1: Przebiegi czasowe 
    Adaptacja kodu z labu 1
'''
fig1, axs1 = plt.subplots(2, 2, figsize=(13, 8))
fig1.suptitle('Zadanie 1: Przebiegi czasowe sygnałów', fontsize=14, fontweight='bold')

for ax, ch, label in zip(axs1.flat, channels, labels):
    ax.scatter(t * 1e3, ch, s=1, alpha=0.6, label=f'{label}(t)')
    ax.set_title(f'Kanał {label}')
    ax.set_xlabel('Czas [ms]')
    ax.set_ylabel('Napięcie [V]')
    ax.legend(loc='upper right', markerscale=4)
    ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout(pad=2.0, w_pad=2.5, h_pad=3.5)
plt.savefig('zad1_przebiegi.png', dpi=150, bbox_inches='tight')

'''
Zadanie 2: Widmo amplitudowe i fazowe (DFT)
    Wyświetlamy tylko przedział 0 … F_MAX_DISPLAY Hz
    Resztę na chwilę obecną możemy pominąć.
'''
dft_results = []
fig2, axs2 = plt.subplots(4, 2, figsize=(14, 14))
fig2.suptitle(f'Zadanie 2: Widmo amplitudowe i fazowe (DFT)\n'
              f'Pokazano 0–{F_MAX_DISPLAY/1e3:.1f} kHz z {fs/2/1e3:.0f} kHz dostępnych',
              fontsize=13, fontweight='bold')

for i, (ch, label) in enumerate(zip(channels, labels)):
    X, amp_pos, phase_pos = compute_dft(ch)
    dft_results.append(X)

    f_plot   = freqs_pos[dft_mask_plot]
    amp_plot = amp_pos[dft_mask_plot]
    phi_plot = phase_pos[dft_mask_plot]

    # Widmo amplitudowe
    ax_a = axs2[i, 0]
    markerline, stemlines, baseline = ax_a.stem(
        f_plot / 1e3, amp_plot, basefmt='k-', linefmt='C0-', markerfmt='C0o')
    markerline.set_markersize(3)
    plt.setp(stemlines, linewidth=0.8)
    ax_a.set_title(f'Widmo amplitudowe – {label}')
    ax_a.set_ylabel('Amplituda [V]')
    ax_a.grid(True, linestyle='--', alpha=0.5)

    # Adnotacja: najsilniejsza składowa (poza DC)
    idx_peak = np.argmax(amp_plot[1:]) + 1
    ax_a.annotate(
        f'{f_plot[idx_peak]/1e3:.2f} kHz\n{amp_plot[idx_peak]:.3f} V',
        xy=(f_plot[idx_peak]/1e3, amp_plot[idx_peak]),
        xytext=(10, 5), textcoords='offset points',
        fontsize=7, color='C0',
        arrowprops=dict(arrowstyle='->', color='C0', lw=0.8))

    # Widmo fazowe – tylko prążki o amplitudzie > 1 % maks.
    sig_mask = amp_plot > (np.max(amp_plot) * 0.01)
    ax_p = axs2[i, 1]
    if sig_mask.sum() > 0:
        markerline2, stemlines2, _ = ax_p.stem(
            f_plot[sig_mask] / 1e3, phi_plot[sig_mask],
            basefmt='k-', linefmt='C1-', markerfmt='C1o')
        markerline2.set_markersize(4)
        plt.setp(stemlines2, linewidth=1.0)
    ax_p.set_title(f'Widmo fazowe – {label}')
    ax_p.set_ylabel('Faza [rad]')
    ax_p.set_ylim(-np.pi - 0.3, np.pi + 0.3)
    ax_p.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax_p.set_yticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    ax_p.grid(True, linestyle='--', alpha=0.5)

for ax in axs2[-1, :]:
    ax.set_xlabel('Częstotliwość [kHz]')

plt.tight_layout(pad=2.0, w_pad=2.0, h_pad=4.0)
plt.savefig('zad2_dft.png', dpi=150, bbox_inches='tight')

'''
Zadanie 3: Widmo amplitudowe (DCT)

Mamy różne typy DCT, tutaj stosujemy typ 2. DCT jest oparte o indeks k, a nie częstotliwość w Hz, można sobie przeliczyć.
Zależność: f_k = k * fs / (2*N)   [Hz]  (dla DCT-2, N próbek), gdzie k=0 odpowiada składowej stałej, k=N-1 odpowiada f = fs/2 - Δf/2.

Wyświetlamy te same k, które odpowiadają 0 … F_MAX_DISPLAY Hz, tj. k ≤ k_max = round(F_MAX_DISPLAY * 2 * N / fs)
'''
k_max = int(np.round(F_MAX_DISPLAY * 2 * N / fs))   # indeks odpowiadający F_MAX_DISPLAY
k_all = np.arange(N)
f_dct = k_all * fs / (2 * N)                         # skala Hz dla osi DCT

fig3, axs3 = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
fig3.suptitle(
    f'Zadanie 3: Widmo amplitudowe (DCT)\n'
    f'Indeksy k ∈ [0, {k_max}], odpowiadające 0–{F_MAX_DISPLAY/1e3:.1f} kHz',
    fontsize=13, fontweight='bold')

for ax, ch, label in zip(axs3, channels, labels):
    X_dct = dct(ch, norm='ortho')              # DCT-2, normalizacja ortogonalna
    amp_dct = np.abs(X_dct[:k_max + 1])

    k_plot = k_all[:k_max + 1]
    markerline, stemlines, _ = ax.stem(
        k_plot, amp_dct, basefmt='k-', linefmt='C2-', markerfmt='C2o')
    markerline.set_markersize(3)
    plt.setp(stemlines, linewidth=0.8)

    # Oznacz dominujący prążek
    idx_peak = np.argmax(amp_dct[1:]) + 1
    ax.annotate(
        f'k={k_plot[idx_peak]}\n≈{f_dct[idx_peak]:.0f} Hz',
        xy=(k_plot[idx_peak], amp_dct[idx_peak]),
        xytext=(8, 4), textcoords='offset points',
        fontsize=7, color='C2',
        arrowprops=dict(arrowstyle='->', color='C2', lw=0.8))

    ax.set_title(f'DCT – {label}')
    ax.set_ylabel('Amplituda DCT\n[j. norm.]')
    ax.grid(True, linestyle='--', alpha=0.5)

# Podwójna oś X: górna w Hz, dolna w indeksie k
axs3[-1].set_xlabel('Indeks składowej k  (oś dolna)')
sec_ax = axs3[-1].secondary_xaxis(
    'bottom',
    functions=(lambda k: k * fs / (2 * N) / 1e3,
               lambda f_khz: f_khz * 1e3 * 2 * N / fs))
# Ręczna alternatywa: dodajemy tylko opis z przelicznikiem
axs3[-1].set_xlabel(f'Indeks k    [przelicznik: f [Hz] = k × {fs/(2*N):.2f}]')

plt.tight_layout(pad=2.0, h_pad=4.0)
plt.savefig('zad3_dct.png', dpi=150, bbox_inches='tight')


''' Rekonstrukcja z sumy cosinusoid, wyników DFT (zadanie dla chętnych) '''
def reconstruct_signal(X_fft, t, N, freqs):
    """Odtwarza sygnał ze składowych DFT (suma cosinusoid)."""
    rec = np.zeros_like(t, dtype=float)
    amp   = np.abs(X_fft) / N
    phase = np.angle(X_fft)
    rec  += amp[0]                        # składowa stała (DC)
    for k in range(1, N // 2):
        rec += 2 * amp[k] * np.cos(2 * np.pi * freqs[k] * t + phase[k])
    return rec

fig4, axs4 = plt.subplots(2, 2, figsize=(13, 8))
fig4.suptitle('Rekonstrukcja sygnału jako suma cosinusoid (DFT)',
              fontsize=13, fontweight='bold')

for ax, ch, X, label in zip(axs4.flat, channels, dft_results, labels):
    rec = reconstruct_signal(X, t, N, freqs)
    err_rms = np.sqrt(np.mean((ch - rec) ** 2))

    ax.plot(t * 1e3, ch,  'b-',  alpha=0.55, linewidth=2.5, label='Oryginał')
    ax.plot(t * 1e3, rec, 'r--', linewidth=1.5, label=f'Rekonstrukcja (RMS err={err_rms:.2e} V)')
    ax.set_title(f'Kanał {label}')
    ax.set_xlabel('Czas [ms]')
    ax.set_ylabel('Napięcie [V]')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout(pad=2.0, w_pad=2.5, h_pad=3.5)
plt.savefig('zad4_rekonstrukcja.png', dpi=150, bbox_inches='tight')

plt.show()
print("\nWszystkie wykresy zapisano do plików PNG.")
