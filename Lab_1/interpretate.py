import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

df = pd.read_csv("data.csv", usecols = [1, 2, 3, 4, 5, 6])
df['Increment'] = df['Increment'].ffill()
df['Start'] = df['Increment'].cumsum()

fig = plt.figure(figsize=(12, 16)) 
gs = gridspec.GridSpec(4, 2)

ax_main = fig.add_subplot(gs[0:2, :])
channels = ['CH1', 'CH2', 'CH3', 'CH4']
colors = ['blue', 'green', 'black', 'red']

for chan, col in zip(channels, colors):
    ax_main.scatter(df['Start'], df[chan], marker='.', linestyle='-', color=col, label=chan)
ax_main.set_title('Zbiór wszystkich odczytów', fontsize=14)
ax_main.grid(True)
ax_main.legend()

coords = [(2, 0), (2, 1), (3, 0), (3, 1)]

for i, (chan, col) in enumerate(zip(channels, colors)):
    ax = fig.add_subplot(gs[coords[i]])
    ax.scatter(df['Start'], df[chan], marker='.', linestyle='-', color=col)
    ax.set_title(f'Zbiór odczytów dla {chan}', fontsize=12)
    ax.set_xlabel('delta [s]')
    ax.grid(True)

plt.subplots_adjust(hspace=0.6, wspace=0.3, top=0.92, bottom=0.08)
plt.show()
