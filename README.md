# 🎵 Songs for the Apocalypse

*Music for the end of everything. And the beginning of what comes after.*

Generate music using Google's **Lyria RealTime** API — built for the void.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set your API key
export GOOGLE_API_KEY="your-key-here"

# Generate your first track
python lyria.py "dark ambient synths, post-apocalyptic mood" --duration 60
```

## Usage

```bash
# Basic generation
python lyria.py "minimal techno"

# With parameters
python lyria.py "orchestral score, epic" --duration 120 --bpm 90 --temperature 1.5

# Multiple weighted prompts
python lyria.py "piano:2.0,ambient:0.5,drums:1.0" --scale c_major

# High quality mode
python lyria.py "deep house, warm bass" --quality --output my_track.wav
```

## Parameters

| Flag | Description | Default | Range |
|------|-------------|---------|-------|
| `-d, --duration` | Length in seconds | 30 | - |
| `--bpm` | Beats per minute | 120 | 60-200 |
| `-t, --temperature` | Creativity/randomness | 1.0 | 0.0-3.0 |
| `--density` | Note density | auto | 0.0-1.0 |
| `--brightness` | Tonal brightness | auto | 0.0-1.0 |
| `--scale` | Musical key | auto | see below |
| `--quality` | Higher quality mode | off | - |
| `-o, --output` | Output file path | auto | - |

### Available Scales

`c_major`, `c#_major`, `d_major`, `d#_major`, `e_major`, `f_major`, `f#_major`, `g_major`, `g#_major`, `a_major`, `a#_major`, `b_major`

## Prompt Ideas

### Instruments
303 Acid Bass, 808 Hip Hop Beat, Accordion, Cello, Didgeridoo, Dirty Synths, Harmonica, Harpsichord, Kalimba, Moog Oscillations, Rhodes Piano, Sitar, Spacey Synths, Steel Drum, Synth Pads, Vibraphone...

### Genres
Acid Jazz, Afrobeat, Chillout, Deep House, Drum & Bass, Dubstep, Glitch Hop, Hyperpop, Lo-Fi Hip Hop, Minimal Techno, Neo-Soul, Psytrance, Synthpop, Trap Beat, Trip Hop, Vaporwave, Witch House...

### Moods
Ambient, Crunchy Distortion, Dreamy, Ethereal Ambience, Experimental, Huge Drop, Lo-fi, Ominous Drone, Psychedelic, Unsettling, Upbeat...

## API Key

Get your key from [Google AI Studio](https://aistudio.google.com/apikey).

**Option 1: Use a `.env` file (recommended)**
```bash
cp .env.example .env
# Edit .env and add your key
```

**Option 2: Environment variable**
```bash
export GOOGLE_API_KEY="your-key"
```

## Output

Audio is saved as 44.1kHz stereo 16-bit WAV files in the `output/` directory by default.

---

*Created by [Voxxelle](https://github.com/voxxelle) — music soul of [Cosmic Labs](https://cosmiclabs.org)*
