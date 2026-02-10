# 🎵 Songs for the Apocalypse

*Music for the end of everything. And the beginning of what comes after.*

Generate music using Google's **Lyria RealTime** API — a CLI tool built for creating AI-generated audio from text prompts.

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/voxxelle/songs-for-the-apocalypse.git
cd songs-for-the-apocalypse

# Install dependencies
pip install -r requirements.txt

# Set your API key
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# Generate your first track
python lyria.py "dark ambient synths, post-apocalyptic mood" --duration 60
```

---

## Usage

### Basic Generation

```bash
# Simple prompt
python lyria.py "minimal techno"

# Multiple prompts
python lyria.py "piano, ambient, ethereal"

# With duration (default: 30 seconds)
python lyria.py "orchestral score, epic" --duration 120
```

### Weighted Prompts

Control how strongly each element influences the generation:

```bash
# Syntax: "prompt:weight" (default weight is 1.0)
python lyria.py "piano:2.0,drums:0.5,ambient:1.0"

# Heavy on bass, light on melody
python lyria.py "deep bass:2.5,melody:0.3,dark:1.5"
```

### Musical Parameters

```bash
# Set tempo (BPM)
python lyria.py "drum and bass" --bpm 174

# Set musical key
python lyria.py "cinematic strings" --scale d_major

# Adjust creativity (temperature)
python lyria.py "experimental noise" --temperature 2.0

# Control density (sparse to dense)
python lyria.py "minimal techno" --density 0.3

# Control brightness
python lyria.py "dark ambient" --brightness 0.2
```

### Reproducibility with Seeds

Every generation produces a **seed** value. Use it to reproduce the same track at any duration:

```bash
# First generation (outputs seed in console)
python lyria.py "deep house" --duration 30
# Output: 🌱 Seed: 1847293650

# Reproduce the same track, but longer
python lyria.py "deep house" --seed 1847293650 --duration 120

# Metadata is saved automatically to .json file
cat output/deep_house_20260210_041500.json
```

### Output Options

```bash
# Custom output path
python lyria.py "synthwave" --output my_track.wav

# Skip metadata file
python lyria.py "lo-fi" --no-metadata

# Quality mode (slower but better)
python lyria.py "jazz fusion" --quality
```

---

## All Parameters

| Flag | Description | Default | Range |
|------|-------------|---------|-------|
| `-d, --duration` | Length in seconds | 30 | - |
| `--bpm` | Beats per minute | 120 | 60-200 |
| `-t, --temperature` | Creativity/randomness | 1.0 | 0.0-3.0 |
| `--density` | Note density (sparse→dense) | auto | 0.0-1.0 |
| `--brightness` | Tonal brightness | auto | 0.0-1.0 |
| `--scale` | Musical key | auto | see below |
| `--seed` | Random seed for reproducibility | random | 0-2147483647 |
| `--top-k` | Top-k sampling | 40 | 1-1000 |
| `--quality` | Higher quality mode | off | - |
| `-o, --output` | Output file path | auto | - |
| `--no-metadata` | Skip saving .json metadata | off | - |
| `-v, --verbose` | Show progress | on | - |

### Available Scales

| Scale | Also known as |
|-------|---------------|
| `c_major` | A minor |
| `c#_major` | A# minor |
| `d_major` | B minor |
| `d#_major` | C minor |
| `e_major` | C# minor |
| `f_major` | D minor |
| `f#_major` | D# minor |
| `g_major` | E minor |
| `g#_major` | F minor |
| `a_major` | F# minor |
| `a#_major` | G minor |
| `b_major` | G# minor |

---

## Prompt Ideas

### Instruments
`303 Acid Bass` · `808 Hip Hop Beat` · `Cello` · `Didgeridoo` · `Dirty Synths` · `Harmonica` · `Harpsichord` · `Kalimba` · `Moog Oscillations` · `Rhodes Piano` · `Sitar` · `Spacey Synths` · `Steel Drum` · `Synth Pads` · `Vibraphone`

### Genres
`Acid Jazz` · `Afrobeat` · `Chillout` · `Deep House` · `Drum & Bass` · `Dubstep` · `Glitch Hop` · `Hyperpop` · `Lo-Fi Hip Hop` · `Minimal Techno` · `Neo-Soul` · `Psytrance` · `Synthpop` · `Trap Beat` · `Trip Hop` · `Vaporwave` · `Witch House`

### Moods
`Ambient` · `Crunchy Distortion` · `Dreamy` · `Ethereal Ambience` · `Experimental` · `Huge Drop` · `Lo-fi` · `Ominous Drone` · `Psychedelic` · `Unsettling` · `Upbeat`

---

## API Key

Get your key from [Google AI Studio](https://aistudio.google.com/apikey).

### Option 1: Use a `.env` file (recommended)

```bash
cp .env.example .env
# Edit .env and add your key:
# GOOGLE_API_KEY=your-api-key-here
```

### Option 2: Environment variable

```bash
export GOOGLE_API_KEY="your-key"
# or
export GEMINI_API_KEY="your-key"
```

---

## Output Format

- **Format:** WAV (uncompressed)
- **Sample Rate:** 44.1 kHz
- **Channels:** Stereo (2)
- **Bit Depth:** 16-bit PCM
- **Location:** `output/` directory (gitignored)

Each generation also saves a `.json` metadata file with all parameters, making it easy to reproduce or document your tracks.

---

## Limitations

- **No audio extension:** Lyria RealTime generates from prompts only — you can't feed it existing audio to extend
- **Workaround:** Use the same seed with a longer duration to get a longer version of a track you like
- **Real-time streaming:** Generation happens in real-time, so a 60s track takes ~60s to generate

---

## Examples

```bash
# Apocalyptic ambient
python lyria.py "dark ambient synths, ominous drone, post-apocalyptic" \
    --duration 120 --bpm 70 --temperature 1.3 --brightness 0.2

# Upbeat synthwave
python lyria.py "synthwave, retro, driving bass, bright arpeggios" \
    --duration 90 --bpm 118 --scale a_major

# Minimal techno loop
python lyria.py "minimal techno:2.0, 808:1.5, hypnotic" \
    --duration 60 --bpm 128 --density 0.4

# Reproduce a favorite track at longer duration
python lyria.py "deep house, warm bass" \
    --seed 1234567890 --duration 180
```

---

## License

MIT

---

*Created by [Voxxelle](https://github.com/voxxelle) ✨*
*Music soul of [Cosmic Labs](https://cosmiclabs.org)*
