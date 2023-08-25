import numpy as np
import pyaudio
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from pygame import mixer
from collections import deque

# Initialize pygame mixer
mixer.init()

# Constants
RATE = 44100
CHUNK = 1024 
DURATION = 1  # 10 seconds
NUM_CHUNKS = int(DURATION * RATE / CHUNK)
MIN_PITCH = 50
MAX_PITCH = 450
THRESHOLD = 0.01

# Initialize pyaudio instance
audio = pyaudio.PyAudio()

# Initialize x-values for pitch plot
times = np.linspace(0, DURATION, NUM_CHUNKS)

# Initialize pitch storage
pitches = deque([0] * NUM_CHUNKS, maxlen=NUM_CHUNKS)

def get_pitch(data, rate, THRESHOLD=0.01):
    # Get the spicy stuff
    data = np.frombuffer(data, dtype=np.int16)

    # Too quiet? Get out of here!
    min_volume = min_volume_scale.get()
    volume = np.mean(np.abs(data))
    
    if volume < min_volume:
        return None
        
    # Check if sound is too quiet
    rms = np.sqrt(np.mean(data**2))
    if rms < THRESHOLD:
        return None

    # Compute FFT and magnitudes
    w = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(w), 1.0/rate)
    magnitudes = np.abs(w)

    # Ignore data outside of MIN_PITCH and MAX_PITCH
    valid_indices = np.where((freqs > MIN_PITCH) & (freqs < MAX_PITCH))
    valid_freqs = freqs[valid_indices]
    valid_magnitudes = magnitudes[valid_indices]

    # Find the most prominent pitch
    idx = np.argmax(valid_magnitudes)
    freq = valid_freqs[idx]
    
    return abs(freq)

def start_stream(device_index):
    global stream
    try:
        if stream and stream.is_active():
            stream.stop_stream()
            stream.close()
        stream = audio.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=RATE,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=CHUNK,
                        stream_callback=callback)
        stream.start_stream()
    except Exception as e:
        print(f"Error starting stream: {e}")

def on_mic_change(event):
    device_index = devices[mic_var.get()]
    start_stream(device_index)

def on_min_volume_change(val):
    min_volume = float(val)
    # Update any other elements if needed

def callback(in_data, frame_count, time_info, status):
    global min_volume, current_volume_var, canvas_pitch, canvas_spectrogram, ax_pitch, ax_spectrogram, lines_pitch, spectrogram_data, pitches

    try:
        # Convert byte data to numpy array
        signal = np.frombuffer(in_data, dtype=np.int16)
        
        # Calculate volume
        volume = np.mean(np.abs(signal))
        volume_percentage = (volume / 32767) * 300  # Convert to percentage (32767 is the max value for int16)
        # Update the bar's height
        bar_current_volume[0].set_height(volume_percentage)
        # Redraw the canvas to reflect the change
        canvas_current_volume.draw()

        if min_volume < volume_percentage:
            return (in_data, pyaudio.paContinue)  # Exit the callback early

        pitch = get_pitch(in_data, RATE)
        
        if pitch is None:
            return (in_data, pyaudio.paContinue)  # Exit the callback early

        # If pitch is not None, continue with the rest of the logic
        pitches.append(pitch)

        if len(pitches) > NUM_CHUNKS:
            pitches.popleft()

        # Update current pitch readout
        note, note_freq = get_nearest_note(pitch)
        current_pitch_var.set(f"Current Pitch: {note} ({note_freq:.2f} Hz)")

        min_pitch = float(min_pitch_entry.get())
        lines_pitch.set_xdata(times)
        lines_pitch.set_ydata(pitches)
        ax_pitch.axhline(y=min_pitch, color='r', linestyle='-')
        canvas_pitch.draw()

        # Update spectrogram
        spectrogram_data = np.roll(spectrogram_data, -1, axis=1)
        spectrogram_data[:, -1] = np.abs(np.fft.fft(np.frombuffer(in_data, dtype=np.int16)))[:CHUNK // 2]
        ax_spectrogram.imshow(np.log(spectrogram_data + 1e-10), aspect='auto', origin='lower')
        canvas_spectrogram.draw()

        if pitch < min_pitch:
            mixer.Sound('se_graze.wav').play()
    except Exception as e:
        print(f"Error in GUI update: {e}")

    return (in_data, pyaudio.paContinue)

def get_nearest_note(frequency):
    # Calculate the difference between the given frequency and all notes
    differences = {note: abs(freq - frequency) for note, freq in NOTES.items()}
    # Find the note with the smallest difference
    nearest_note = min(differences, key=differences.get)
    return nearest_note, NOTES[nearest_note]

root = tk.Tk()
root.title("Pitch Tracker and Spectrogram")

NOTES = {
    "E2": 82.41,
    "F2": 87.31,
    "F#2/Gb2": 92.50,
    "G2": 98.00,
    "G#2/Ab2": 103.83,
    "A2": 110.00,
    "A#2/Bb2": 116.54,
    "B2": 123.47,
    "C3": 130.81,
    "C#3/Db3": 138.59,
    "D3": 146.83,
    "D#3/Eb3": 155.56,
    "E3": 164.81,
    "F3": 174.61,
    "F#3/Gb3": 185.00,
    "G3": 196.00,
    "G#3/Ab3": 207.65,
    "A3": 220.00,
    "A#3/Bb3": 233.08,
    "B3": 246.94,
    "C4": 261.63,
    "C#4/Db4": 277.18,
    "D4": 293.66,
    "D#4/Eb4": 311.13,
    "E4": 329.63,
    "F4": 349.23,
    "F#4/Gb4": 369.99,
    "G4": 392.00,
    "G#4/Ab4": 415.30,
    "A4": 440.00,
    "A#4/Bb4": 466.16,
    "B4": 493.88,
    "C5": 523.25,
    "C#5/Db5": 554.37,
    "D5": 587.33,
    "D#5/Eb5": 622.25,
    "E5": 659.26,
    "F5": 698.46,
    "F#5/Gb5": 739.99,
    "G5": 783.99,
    "G#5/Ab5": 830.61,
    "A5": 880.00,
    "A#5/Bb5": 932.33,
    "B5": 987.77,
    "C6": 1046.50,
    "C#6/Db6": 1108.73,
    "D6": 1174.66,
    "D#6/Eb6": 1244.51,
    "E6": 1318.51,
    "F6": 1396.91,
    "F#6/Gb6": 1479.98,
    "G6": 1567.98,
    "G#6/Ab6": 1661.22,
    "A6": 1760.00,
    "A#6/Bb6": 1864.66,
    "B6": 1975.53,
    "C7": 2093.00,
    "C#7/Db7": 2217.46,
    "D7": 2349.32,
    "D#7/Eb7": 2489.02,
    "E7": 2637.02
}

# GUI Components

# Dropdown for microphone selection
devices = {audio.get_device_info_by_index(i)["name"]: i for i in range(audio.get_device_count()) if audio.get_device_info_by_index(i)["maxInputChannels"] > 0}
default_device_info = audio.get_default_input_device_info()
mic_var = tk.StringVar(root, value=default_device_info["name"])
mic_dropdown = ttk.Combobox(root, textvariable=mic_var, values=list(devices.keys()), width=50)
mic_dropdown.pack(pady=10)
mic_dropdown.bind("<<ComboboxSelected>>", on_mic_change)

# Entry for minimum pitch
min_pitch_label = tk.Label(root, text="Minimum Pitch:")
min_pitch_label.pack(pady=5)
min_pitch_entry = tk.Entry(root)
min_pitch_entry.pack(pady=5)
min_pitch_entry.insert(0, "180")

# Current pitch readout
current_pitch_var = tk.StringVar(root, value="Current Pitch: ")
current_pitch_label = tk.Label(root, textvariable=current_pitch_var)
current_pitch_label.pack(pady=5)

# Current volume plot
fig_current_volume, ax_current_volume = plt.subplots(figsize=(2, 4))
bar_current_volume = ax_current_volume.bar(0, 0, width=2, color='g')
# ax_current_volume.set_xlim(-1, 1)
ax_current_volume.set_title("Current Volume")
ax_current_volume.set_ylim(0, 100)
ax_current_volume.set_xticks([])
canvas_current_volume = FigureCanvasTkAgg(fig_current_volume, master=root)
canvas_current_volume.draw()
canvas_current_volume.get_tk_widget().pack(side=tk.LEFT, fill=tk.Y)

# Scale for setting minimum volume
#min_volume_label = tk.Label(root, text="Minimum Volume (%):")
# min_volume_label.pack(pady=5)
min_volume = 10
min_volume_scale = tk.Scale(root, from_=100, to_=0, resolution=1, orient=tk.VERTICAL)
min_volume_scale.set(min_volume)  # Default value
min_volume_scale.pack(side=tk.LEFT, fill=tk.Y)
min_volume_scale.bind("<Motion>", lambda event: on_min_volume_change(min_volume_scale.get()))

# Pitch plot
fig_pitch, ax_pitch = plt.subplots()
lines_pitch, = ax_pitch.plot(times, [0]*len(times), lw=2)
ax_pitch.set_title("Pitch Over Time")
ax_pitch.set_xlabel("Time (s)")
ax_pitch.set_ylabel("Pitch (Hz)")
ax_pitch.set_ylim(MIN_PITCH, MAX_PITCH)
ax_pitch.set_xlim(0, DURATION)
canvas_pitch = FigureCanvasTkAgg(fig_pitch, master=root)
canvas_pitch.draw()
canvas_pitch.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Spectrogram plot
fig_spectrogram, ax_spectrogram = plt.subplots()
spectrogram_data = np.zeros((CHUNK // 2, NUM_CHUNKS))
ax_spectrogram.set_title("Spectrogram")
ax_spectrogram.set_xlabel("Time (s)")
ax_spectrogram.set_ylabel("Frequency (Hz)")
ax_spectrogram.set_xlim(0, DURATION)
ax_spectrogram.set_ylim(MIN_PITCH, MAX_PITCH)
ax_spectrogram.imshow(np.log(spectrogram_data + 1e-10), aspect='auto', origin='lower', cmap='inferno', interpolation='bilinear')
canvas_spectrogram = FigureCanvasTkAgg(fig_spectrogram, master=root)
canvas_spectrogram.draw()
canvas_spectrogram.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Start the audio stream with the default device
stream = None
start_stream(default_device_info["index"])

root.mainloop()

# Cleanup
if stream:
    stream.stop_stream()
    stream.close()
audio.terminate()
