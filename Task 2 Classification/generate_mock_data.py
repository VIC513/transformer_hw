from __future__ import annotations

import argparse
import random
from pathlib import Path


try:
    import pretty_midi
except Exception as e:  # pragma: no cover
    pretty_midi = None
    _pretty_midi_err = e


from dataset import GENRES


def _require_pretty_midi():
    if pretty_midi is None:
        raise ImportError(
            "pretty_midi is required for mock data generation. Install with `pip install pretty_midi`."
        ) from _pretty_midi_err


def make_random_midi(
    out_path: Path,
    seed: int,
    min_notes: int = 80,
    max_notes: int = 260,
    min_pitch: int = 36,
    max_pitch: int = 96,
):
    _require_pretty_midi()
    rng = random.Random(seed)

    midi = pretty_midi.PrettyMIDI(initial_tempo=rng.uniform(80.0, 160.0))
    program = rng.choice([0, 24, 40, 41, 56, 73])
    inst = pretty_midi.Instrument(program=program, is_drum=False)

    n_notes = rng.randint(min_notes, max_notes)
    t = 0.0
    for _ in range(n_notes):
        dt = rng.uniform(0.03, 0.25)
        dur = rng.uniform(0.05, 0.45)
        pitch = rng.randint(min_pitch, max_pitch)
        velocity = rng.randint(50, 110)
        start = t + rng.uniform(0.0, 0.02)
        end = start + dur
        inst.notes.append(
            pretty_midi.Note(velocity=int(velocity), pitch=int(pitch), start=float(start), end=float(end))
        )
        t += dt

    midi.instruments.append(inst)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(out_path))


def main():
    parser = argparse.ArgumentParser(description="Generate mock MIDI genre dataset")
    parser.add_argument("--root", type=str, default="./data/genres")
    parser.add_argument("--per_genre", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    made = 0
    skipped = 0
    for genre in GENRES:
        gdir = root / genre
        gdir.mkdir(parents=True, exist_ok=True)
        for i in range(int(args.per_genre)):
            out_path = gdir / f"{genre}_{i:03d}.mid"
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue
            seed = int(args.seed) + (hash(genre) % 10_000) + i
            make_random_midi(out_path, seed=seed)
            made += 1

    print(f"root: {root.resolve()}")
    print(f"generated: {made}, skipped: {skipped}")


if __name__ == "__main__":
    main()

