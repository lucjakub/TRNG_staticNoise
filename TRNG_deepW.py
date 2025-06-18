import numpy as np
import sys
import os
from scipy.io import wavfile
import hashlib
import matplotlib.pyplot as plt

def load_audio(file_path):
    sample_rate, data = wavfile.read(file_path)
    if len(data.shape) > 1:
        data = data[:, 0]  # użycie pierwszego kanału, jesli zrodlo jest w stereo

    return sample_rate, data

def display_source_samples(data):
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=100, color='skyblue', edgecolor='black')
    plt.title("rozkład próbek ze źródła")
    plt.xlabel("wartość próbki")
    plt.ylabel("częstotliwość")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def display_numbers(data):
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=100, density=True, color='black', edgecolor='black')
    plt.title("rozkład zmiennych losowych")
    plt.xlabel("wartość próbki")
    plt.ylabel("prawdopodobieństwo wystąpienia próbki")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def calculate_shannon_entropy(samples):
    #Obliczenie entropii dla probek ze zrodła
    samples = samples.astype(np.float32)
    min_val = np.min(samples)
    max_val = np.max(samples)
    if max_val == min_val:
        return 0.0  # wszystkie wartosci sa takie same - entropia = 0
    normalized_samples = (samples - min_val) / (max_val - min_val)
    hist, _ = np.histogram(normalized_samples, bins=256, density=False)
    total = np.sum(hist)
    if total == 0:
        return 0.0 
    probs = hist / total
    probs = probs[probs > 0]
    shannon_entropy = -np.sum(probs * np.log2(probs))

    return shannon_entropy

def hash_chunk_to_number(chunk):
    hasher = hashlib.sha256()
    hasher.update(chunk.tobytes())
    digest = hasher.digest()[:4]  # Use first 4 bytes
    save_to_bin(digest.hex(), "sha", "a")
    uint_val = int.from_bytes(digest, 'big')
    return uint_val / 0xFFFFFFFF  # Normalize to [0, 1]

def generate_numbers_from_float(input_float):
    # Zamień float na string i pobierz część po przecinku
    decimal = str(input_float).split(".")[1][:8]  # Tylko pierwsze 8 cyfr
    return int(''.join(['1' if int(digit) % 2 != 0 else '0' for digit in decimal]), 2)

def generate_random_numbers_from_chunks(samples, sample_rate, seconds_per_chunk, num_numbers):
    #generowanie liczb z chunków
    samples_per_chunk = int(sample_rate * seconds_per_chunk)
    random_numbers = []

    for i in range(num_numbers):
        start = i * samples_per_chunk
        end = start + samples_per_chunk
        if end > len(samples):
            raise Exception(f"Source file to short requestet seconds_per_chunk: {seconds_per_chunk} and num_numbers: {num_numbers}")
        chunk = samples[start:end]
        number = hash_chunk_to_number(chunk)
        random_numbers.append(generate_numbers_from_float(number))

    return random_numbers

def save_to_bin(data, file_name, mode = 'w'):
    with open(f"{file_name}.bin", mode) as f:
        f.write(str(data) + '\n')

def cleanup(paths):
    for p in paths:
        if os.path.exists(p):
            os.remove(p)

def main(file_path, seconds_per_chunk, num_random_numbers):

    sample_rate, audio_samples = load_audio(file_path)

    display_source_samples(audio_samples)

    save_to_bin(audio_samples.tolist(), "source")

    entropy_value = calculate_shannon_entropy(audio_samples)
    print(f"Shannon entropy for provided source: {entropy_value} \n")

    random_numbers = generate_random_numbers_from_chunks(audio_samples, sample_rate, seconds_per_chunk, num_random_numbers)
    print(f"Generated Random Numbers: {random_numbers}")

    save_to_bin(random_numbers, "post", "w")

    output_entropy = calculate_shannon_entropy(np.array(random_numbers))
    print(f"\n Entropy of the generated output: {output_entropy}")

    display_numbers(random_numbers)

if __name__ == "__main__":
    cleanup(['post.bin', 'sha.bin', 'source.bin'])
    if len(sys.argv) != 4:
        print("Invalid usage!\nProper usage: python3 <file_path.wav> <seconds_per_chunk> <num_random_numbers>")
        sys.exit()
    main(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]))