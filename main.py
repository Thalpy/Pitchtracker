import numpy as np
import pyaudio
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from pygame import mixer
from collections import deque
import threading

class PitchTracker:
    def __init__(self):
        # Constants
        self.RATE = 44100
        self.CHUNK = 1024 
        self.DURATION = 1
        self.NUM_CHUNKS = int(self.DURATION * self.RATE / self.CHUNK)
        self.MIN_PITCH = 50
        self.MAX_PITCH = 450
        self.THRESHOLD = 0.01

        # Initialize pyaudio instance
        self.audio = pyaudio.PyAudio()

        # Initialize x-values for pitch plot
        self.times = np.linspace(0, self.DURATION, self.NUM_CHUNKS)

        # Initialize pitch storage
        self.pitches = deque([0] * self.NUM_CHUNKS, maxlen=self.NUM_CHUNKS)

        # Initialize pygame mixer
        mixer.init()

        # GUI Components
        self.root = tk.Tk()
        self.root.title("Pitch Tracker and Spectrogram")

        self.NOTES = {
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

        def on_mic_change(event):
            device_index = devices[mic_var.get()]
            start_stream(device_index)

        def on_min_volume_change(val):
            self.min_volume= float(val)
            # Update any other elements if needed

        # GUI Components

        # Dropdown for microphone selection
        devices = {self.audio.get_device_info_by_index(i)["name"]: i for i in range(self.audio.get_device_count()) if self.audio.get_device_info_by_index(i)["maxInputChannels"] > 0}
        default_device_info = self.audio.get_default_input_device_info()
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
        #bar_current_volume = ax_current_volume.bar(0, 0, width=2, color='g')
        # ax_current_volume.set_xlim(-1, 1)
        ax_current_volume.set_title("Current Volume")
        ax_current_volume.set_ylim(0, 100)
        ax_current_volume.set_xticks([])
        self.canvas_current_volume = FigureCanvasTkAgg(fig_current_volume, master=root)
        self.canvas_current_volume.draw()
        self.canvas_current_volume.get_tk_widget().pack(side=tk.LEFT, fill=tk.Y)

        # Scale for setting minimum volume
        self.min_volume= 10
        min_volume_scale = tk.Scale(root, from_=100, to_=0, resolution=1, orient=tk.VERTICAL)
        min_volume_scale.set(self.min_volume)  # Default value
        min_volume_scale.pack(side=tk.LEFT, fill=tk.Y)
        min_volume_scale.bind("<Motion>", lambda event: on_min_volume_change(min_volume_scale.get()))

        # Pitch plot
        fig_pitch, self.ax_pitch = plt.subplots()
        self.lines_pitch, = self.ax_pitch.plot(self.times, [0]*len(self.times), lw=2)
        self.ax_pitch.set_title("Pitch Over Time")
        self.ax_pitch.set_xlabel("Time (s)")
        self.ax_pitch.set_ylabel("Pitch (Hz)")
        self.ax_pitch.set_ylim(self.MIN_PITCH, self.MAX_PITCH)
        self.ax_pitch.set_xlim(0, self.DURATION)
        self.canvas_pitch = FigureCanvasTkAgg(fig_pitch, master=root)
        self.canvas_pitch.draw()
        self.canvas_pitch.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Spectrogram plot
        fig_spectrogram, self.ax_spectrogram = plt.subplots()
        self.spectrogram_data = np.zeros((self.CHUNK // 2, self.NUM_CHUNKS))
        self.ax_spectrogram.set_title("Spectrogram")
        self.ax_spectrogram.set_xlabel("Time (s)")
        self.ax_spectrogram.set_ylabel("Frequency (Hz)")
        self.ax_spectrogram.set_xlim(0, self.DURATION)
        self.ax_spectrogram.set_ylim(self.MIN_PITCH, self.MAX_PITCH)
        self.ax_spectrogram.imshow(np.log(self.spectrogram_data + 1e-10), aspect='auto', origin='lower', cmap='inferno', interpolation='bilinear')
        self.canvas_spectrogram = FigureCanvasTkAgg(fig_spectrogram, master=root)
        self.canvas_spectrogram.draw()
        self.canvas_spectrogram.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Start the self.audio stream with the default device
        self.stream = None
        default_device_info = self.audio.get_default_input_device_info()
        self.start_stream(default_device_info["index"])

        self.root.mainloop()

        # Cleanup
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

    def start_stream(self, device_index):
        def threaded_stream():
            try:
                if self.stream and self.stream.is_active():
                    self.stream.stop_stream()
                    self.stream.close()
                self.stream = self.audio.open(format=pyaudio.paInt16,
                                    channels=1,
                                    rate=self.RATE,
                                    input=True,
                                    input_device_index=device_index,
                                    frames_per_buffer=self.CHUNK,
                                    stream_callback=self.callback)
                self.stream.start_stream()
            except Exception as e:
                print(f"Error starting stream: {e}")

        threading.Thread(target=threaded_stream).start()

    def get_pitch(self, data, rate, THRESHOLD=0.01):
        # Get the spicy stuff
        data = np.frombuffer(data, dtype=np.int16)

        # Too quiet? Get out of here!
        volume = np.mean(np.abs(data))
        
        if volume < self.min_volume:
            return None
            
        # Check if sound is too quiet
        rms = np.sqrt(np.mean(data**2))
        if rms < THRESHOLD:
            return None

        # Compute FFT and magnitudes
        w = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(w), 1.0/rate)
        magnitudes = np.abs(w)

        # Ignore data outside of self.MIN_PITCH and self.MAX_PITCH
        valid_indices = np.where((freqs > self.MIN_PITCH) & (freqs < self.MAX_PITCH))
        valid_freqs = freqs[valid_indices]
        valid_magnitudes = magnitudes[valid_indices]

        # Find the most prominent pitch
        idx = np.argmax(valid_magnitudes)
        freq = valid_freqs[idx]
        
        return abs(freq)

    def callback(self, in_data, frame_count, time_info, status):

        try:
            # Convert byte data to numpy array
            signal = np.frombuffer(in_data, dtype=np.int16)
            
            # Calculate volume
            volume = np.mean(np.abs(signal))
            volume_percentage = (volume / 32767) * 300  # Convert to percentage (32767 is the max value for int16)
            # Update the bar's height
            #bar_current_volume[0].set_height(volume_percentage)
            # Redraw the canvas to reflect the change
            self.canvas_current_volume.draw()

            if self.min_volume< volume_percentage:
                return (in_data, pyaudio.paContinue)  # Exit the callback early

            pitch = get_pitch(in_data, self.RATE)
            
            if pitch is None:
                return (in_data, pyaudio.paContinue)  # Exit the callback early

            # If pitch is not None, continue with the rest of the logic
            self.pitches.append(pitch)

            if len(self.pitches) > self.NUM_CHUNKS:
                self.pitches.popleft()

            # Update current pitch readout
            note, note_freq = get_nearest_note(pitch)
            current_pitch_var.set(f"Current Pitch: {note} ({note_freq:.2f} Hz)")

            min_pitch = float(min_pitch_entry.get())
            self.lines_pitch.set_xdata(self.times)
            self.lines_pitch.set_ydata(self.pitches)
            self.ax_pitch.axhline(y=min_pitch, color='r', linestyle='-')
            self.canvas_pitch.draw()

            # Update spectrogram
            self.spectrogram_data = np.roll(self.spectrogram_data, -1, axis=1)
            self.spectrogram_data[:, -1] = np.abs(np.fft.fft(np.frombuffer(in_data, dtype=np.int16)))[:self.CHUNK // 2]
            self.ax_spectrogram.imshow(np.log(self.spectrogram_data + 1e-10), aspect='auto', origin='lower')
            self.canvas_spectrogram.draw()

            if pitch < min_pitch:
                mixer.Sound('se_graze.wav').play()
        except Exception as e:
            print(f"Error in GUI update: {e}")

        return (in_data, pyaudio.paContinue)

    def get_nearest_note(self, frequency):
        # Calculate the difference between the given frequency and all notes
        differences = {note: abs(freq - frequency) for note, freq in self.NOTES.items()}
        # Find the note with the smallest difference
        nearest_note = min(differences, key=differences.get)
        return nearest_note, self.NOTES[nearest_note]
    

root = tk.Tk()
root.title("Pitch Tracker and Spectrogram")


if __name__ == "__main__":
    app = PitchTracker()