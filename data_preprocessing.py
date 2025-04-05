import os
import pretty_midi
import numpy as np
import music21
import json
from functools import lru_cache


"""
    Load MIDI files
"""
def parse_midi_file(file_path):
    # Parse MIDI file using pretty_midi
    try:
        midi_data = pretty_midi.PrettyMIDI(file_path)
        return midi_data
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def collect_midi_paths(root_dir):
    # Recursively collect .mid and .midi files from directory
    midi_paths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.mid', '.midi')):
                midi_paths.append(os.path.join(root, file))
    return midi_paths



"""
    Normalization pitch and velocity
"""
def normalize_pitch(notes, lower=21, upper=108):
    # Limit note pitch to [lower, upper]
    for note in notes:
        if note.pitch < lower:
            note.pitch = lower
        elif note.pitch > upper:
            note.pitch = upper
    return notes

def normalize_velocity(notes):
    # Normalize velocity to [0, 1]
    for note in notes:
        note.velocity = note.velocity / 127.0
    return notes



"""
    Align each note's time position to a fixed time grid
"""
def quantize_notes(notes, ticks_per_beat=4, tempo=120):
    seconds_per_beat = 60.0 / tempo
    for note in notes:
        quantized_start = round(
            (note.start / seconds_per_beat) * ticks_per_beat) / ticks_per_beat
        quantized_end = round(
            (note.end / seconds_per_beat) * ticks_per_beat) / ticks_per_beat
        note.start = quantized_start * seconds_per_beat
        note.end = quantized_end * seconds_per_beat
    return notes


"""
    Convert MIDI to piano roll
    and extract simultaneous notes
"""
def midi_to_piano_roll(midi_data, fs=100):
    piano_rolls = [inst.get_piano_roll(fs=fs) for inst in midi_data.instruments]
    max_time_steps = max(roll.shape[1] for roll in piano_rolls)
    
    piano_roll = np.zeros((128, max_time_steps))

    for roll in piano_rolls:
        # Pad roll to max_time_steps only if needed
        if roll.shape[1] < max_time_steps:
            padded_roll = np.zeros((128, max_time_steps))
            padded_roll[:, :roll.shape[1]] = roll
            roll = padded_roll
        piano_roll = np.maximum(piano_roll, roll)

    return piano_roll



def get_simultaneous_notes(piano_roll, threshold=0):
    # Extract simultaneous notes at each time step
    simultaneous = []
    for t in range(piano_roll.shape[1]):
        pitches = np.where(piano_roll[:, t] > threshold)[0]
        simultaneous.append(list(pitches))
    return simultaneous



"""
    Label chords using music21 and extract chord labels
"""
@lru_cache(maxsize=5000)
def label_chord_cached(pitches_tuple, max_notes=6):
    if not pitches_tuple:
        return "No Chord"
    pitches = list(pitches_tuple)
    if len(pitches) > max_notes:
        pitches = sorted(pitches)[:max_notes]
    notes = [music21.note.Note(p) for p in pitches]
    chord_obj = music21.chord.Chord(notes)
    chord_label = chord_obj.pitchedCommonName
    return chord_label if chord_label else "Unknown Chord"

def label_chords(simultaneous_notes):
    labels = [label_chord_cached(tuple(pitches)) for pitches in simultaneous_notes]
    return labels


"""
    Main function to process MIDI files
"""
def process_midi_file(file_path, pitch_range=(21,108), normalize_vel=True, ticks_per_beat=4, fs=50):
    midi_data = parse_midi_file(file_path)
    if midi_data is None:
        return None

    notes = [note for inst in midi_data.instruments for note in inst.notes]
    notes.sort(key=lambda x: x.start)

    normalize_pitch(notes, pitch_range[0], pitch_range[1])
    if normalize_vel:
        normalize_velocity(notes)

    tempos = midi_data.get_tempo_changes()[1]
    tempo = tempos[0] if len(tempos) > 0 else 120

    quantize_notes(notes, ticks_per_beat, tempo)

    piano_roll = midi_to_piano_roll(midi_data, fs)
    simultaneous_notes = get_simultaneous_notes(piano_roll)
    chords = label_chords(simultaneous_notes)

    return {
        'notes': notes,
        'piano_roll': piano_roll,
        'chord_labels': chords
    }


def process_dataset(root_dir, max_files=None):
    # Process MIDI dataset by iterating over files in root_dir.
    midi_paths = collect_midi_paths(root_dir)
    if max_files:
        midi_paths = midi_paths[:max_files]
    results = []
    for i, path in enumerate(midi_paths):
        print(f"Processing file {i+1}/{len(midi_paths)}: {path}")
        result = process_midi_file(path)
        if result is not None:
            results.append(result)
    return results

"""
    Save results to a JSON file
"""
def save_results_to_file(results, output_file="test_results.json"):
    # Create a summary of results for each processed file.
    summary = []
    for i, result in enumerate(results):
        file_summary = {
            "FileIndex": i,
            "NumberOfNotes": len(result["notes"]),
            "PianoRollShape": result["piano_roll"].shape,
            "NumberOfChordLabels": len(result["chord_labels"]),
            "ChordLabels": result["chord_labels"]
        }
        summary.append(file_summary)

    # Save summary to a JSON file.
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {output_file}")



"""
    Main entry point
"""
if __name__ == "__main__":
    root_dir = "./lakh-midi-clean"
    results = process_dataset(root_dir, max_files=5)  # Test with 5 files
    print(f"Processed {len(results)} files successfully.")

    save_results_to_file(results) # Save for viewing.
