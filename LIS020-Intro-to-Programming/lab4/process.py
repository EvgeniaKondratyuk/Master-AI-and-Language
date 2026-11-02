"""Reorder a recording of multiple speakers by their mean fundamental
frequency (F0), lowest to highest

Usage:
    python3 process.py <input_wav> <output_wav>
"""

import numpy as np
import sys
import librosa
import soundfile as sf 

def extract_clips(data: np.ndarray, intervals: np.ndarray) -> list[np.ndarray]:
  """Slice an audio array into clips using sample indices [start:end].

    Args:
        data: the input audio 
        intervals: array of shape (n, 2), each row a [start, end] pair
            of sample indices, as returned by librosa.effects.split

    Returns:
        A list of audio clips, one per interval, in the given order
  """
  clips = []
  for start, end in intervals:
    clips.append(data[start:end])
  return clips

def mean_f0_per_clip(clips: list[np.ndarray], sample: int) -> list[tuple[np.ndarray, float]]:
  """Compute the mean F0 of each clip, discarding unmeasurable clips

    Uses librosa.pyin to estimate a pitch (F0) curve per clip, then
    averages it with numpy.nanmean, ignoring unvoiced frames. Clips
    whose F0 curve is entirely NaN (no pitch could be detected at all)
    are left out of the result.

    Args:
        clips: list of audio clips to analyze
        sample_rate: sample rate of the audio, as returned by soundfile.read

    Returns:
        A list of (clip, mean_f0) pairs, one per clip where F0 could be
        measured
  """
  results = []
  for clip in clips:
    f0, _, _ = librosa.pyin(clip, sr = sample, fmin=70, fmax=500)
    if np.all(np.isnan(f0)):
      continue
    results.append((clip, np.mean(f0)))
  return results


def main() -> None: 
    """Run the full pipeline: read, split, analyze, sort, and write audio.

    Reads an input WAV file and an output path from the command line,
    splits the recording into clips, computes each clip's mean F0,
    discards clips where F0 could not be measured, reorders the
    remaining clips from lowest to highest F0, concatenates them, and
    writes the result to the output path.

    Exits with status 1 and an informative message if the command-line
    arguments are missing, the input file cannot be read, or the
    output file cannot be written.
    """

    # check the required number of arguments: script name, input path, output path
    if len(sys.argv) != 3:
        print("Usage: python3 process.py <input_wav> <output_wav>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # if reading file fails or the file is missing, unreadable, or not valid audio format
    try:
        data, sample = sf.read(input_file)
    except Exception as e:
        print(f"Error: could not read the input file '{input_file}': {e}")
        sys.exit(1)
    
    # to find and extract the non-silent segments
    intervals = librosa.effects.split(data)
    clips = extract_clips(data, intervals)

    # estimate each clip's pitch and reorder clips from lowest to highest mean F0
    clip_f0_pairs = mean_f0_per_clip(clips, sample)
    sorted_pairs = sorted(clip_f0_pairs, key=lambda pair: pair[1])
    sorted_clips = [clip for clip, _ in sorted_pairs]

    # to join the reordered clips into one continuous audio 
    final_audio = np.concatenate(sorted_clips)

    # if writing fails or the output path is not ivalid or not writable
    try:
        sf.write(output_file, final_audio, sample)
    except Exception as e:
        print(f"Error: could not write the output file '{output_file}': {e}")
        sys.exit(1)

    print(f"Wrote {len(sorted_clips)} clips, sorted by F0, to '{output_file}'.")


if __name__ == "__main__":
    main()