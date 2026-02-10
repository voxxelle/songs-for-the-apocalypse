#!/usr/bin/env python3
"""
🎵 Lyria Music Generator
Generate music using Google's Lyria RealTime API.

Usage:
    python lyria.py "dark ambient synths, post-apocalyptic mood"
    python lyria.py "minimal techno" --duration 60 --bpm 128
    python lyria.py "orchestral score, epic" --output epic.wav --temperature 1.5
    python lyria.py "deep house" --seed 12345 --duration 120

Environment:
    GOOGLE_API_KEY or GEMINI_API_KEY must be set.
"""

import argparse
import asyncio
import json
import os
import random
import sys
import wave
from datetime import datetime
from pathlib import Path

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Error: google-genai package not installed.")
    print("Install with: pip install google-genai")
    sys.exit(1)


# Available scales from the API
SCALES = {
    "c_major": "C_MAJOR_A_MINOR",
    "c#_major": "C_SHARP_MAJOR_A_SHARP_MINOR",
    "d_major": "D_MAJOR_B_MINOR",
    "d#_major": "D_SHARP_MAJOR_C_MINOR",
    "e_major": "E_MAJOR_C_SHARP_MINOR",
    "f_major": "F_MAJOR_D_MINOR",
    "f#_major": "F_SHARP_MAJOR_D_SHARP_MINOR",
    "g_major": "G_MAJOR_E_MINOR",
    "g#_major": "G_SHARP_MAJOR_F_MINOR",
    "a_major": "A_MAJOR_F_SHARP_MINOR",
    "a#_major": "A_SHARP_MAJOR_G_MINOR",
    "b_major": "B_MAJOR_G_SHARP_MINOR",
}

# Seed range per Lyria API docs
SEED_MIN = 0
SEED_MAX = 2_147_483_647


def get_api_key() -> str:
    """Get API key from environment."""
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        print("Error: No API key found.")
        print("Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
        sys.exit(1)
    return key


def parse_prompts(prompt_args: list[str]) -> list[dict]:
    """
    Parse prompt arguments into weighted prompts.
    
    Supports formats:
        "minimal techno"           -> weight 1.0
        "piano:2.0"                -> weight 2.0
        "ambient:0.5,drums:1.5"    -> multiple prompts
    """
    prompts = []
    
    for arg in prompt_args:
        # Split by comma for multiple prompts in one arg
        parts = arg.split(",")
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            # Check for weight syntax "prompt:weight"
            if ":" in part and part.split(":")[-1].replace(".", "").isdigit():
                text, weight_str = part.rsplit(":", 1)
                weight = float(weight_str)
            else:
                text = part
                weight = 1.0
            
            prompts.append({"text": text.strip(), "weight": weight})
    
    return prompts


async def generate_music(
    prompts: list[dict],
    duration_seconds: int = 30,
    bpm: int = 120,
    temperature: float = 1.0,
    density: float | None = None,
    brightness: float | None = None,
    scale: str | None = None,
    seed: int | None = None,
    top_k: int | None = None,
    quality_mode: bool = False,
    output_path: str | None = None,
    save_metadata: bool = True,
    verbose: bool = False,
) -> tuple[Path, dict]:
    """
    Generate music using Lyria RealTime.
    
    Args:
        prompts: List of {"text": str, "weight": float} dicts
        duration_seconds: Length of audio to generate
        bpm: Beats per minute (60-200)
        temperature: Creativity/randomness (0.0-3.0)
        density: Note density (0.0-1.0)
        brightness: Tonal brightness (0.0-1.0)
        scale: Musical scale (e.g., "c_major", "d_major")
        seed: Random seed for reproducibility (0 to 2147483647)
        top_k: Top-k sampling (1-1000, default 40)
        quality_mode: Use QUALITY mode instead of default
        output_path: Output file path (default: auto-generated)
        save_metadata: Save generation params to JSON file
        verbose: Print progress information
        
    Returns:
        Tuple of (Path to audio file, metadata dict)
    """
    api_key = get_api_key()
    
    # Generate seed if not provided
    if seed is None:
        seed = random.randint(SEED_MIN, SEED_MAX)
    
    # Initialize client with v1alpha API version
    client = genai.Client(
        api_key=api_key,
        http_options={"api_version": "v1alpha"}
    )
    
    # Audio parameters (Lyria outputs 44.1kHz stereo 16-bit PCM)
    sample_rate = 44100
    channels = 2
    sample_width = 2  # 16-bit = 2 bytes
    
    # Calculate expected bytes
    expected_bytes = sample_rate * channels * sample_width * duration_seconds
    
    # Collect audio chunks
    audio_data = bytearray()
    start_time = None
    
    if verbose:
        print(f"🎵 Generating {duration_seconds}s of music...")
        print(f"   Prompts: {prompts}")
        print(f"   BPM: {bpm} | Temp: {temperature} | Seed: {seed}")
    
    async def receive_audio(session):
        nonlocal audio_data, start_time
        
        async for message in session.receive():
            if start_time is None:
                start_time = datetime.now()
                
            if hasattr(message, 'server_content') and message.server_content:
                if hasattr(message.server_content, 'audio_chunks'):
                    for chunk in message.server_content.audio_chunks:
                        if hasattr(chunk, 'data') and chunk.data:
                            audio_data.extend(chunk.data)
                            
                            if verbose:
                                elapsed = (datetime.now() - start_time).total_seconds()
                                generated = len(audio_data) / (sample_rate * channels * sample_width)
                                print(f"\r   Generated: {generated:.1f}s / {duration_seconds}s", end="", flush=True)
                            
                            # Stop when we have enough audio
                            if len(audio_data) >= expected_bytes:
                                return
            
            # Tiny sleep to prevent blocking
            await asyncio.sleep(1e-6)
    
    try:
        async with client.aio.live.music.connect(model="models/lyria-realtime-exp") as session:
            # Create receive task
            receive_task = asyncio.create_task(receive_audio(session))
            
            # Build config
            config_kwargs = {
                "bpm": bpm,
                "temperature": temperature,
                "seed": seed,
            }
            
            if density is not None:
                config_kwargs["density"] = density
            if brightness is not None:
                config_kwargs["brightness"] = brightness
            if top_k is not None:
                config_kwargs["top_k"] = top_k
            if scale and scale.lower() in SCALES:
                scale_enum = getattr(types.Scale, SCALES[scale.lower()], None)
                if scale_enum:
                    config_kwargs["scale"] = scale_enum
            if quality_mode:
                config_kwargs["music_generation_mode"] = types.MusicGenerationMode.QUALITY
            
            # Set prompts
            weighted_prompts = [
                types.WeightedPrompt(text=p["text"], weight=p["weight"])
                for p in prompts
            ]
            await session.set_weighted_prompts(prompts=weighted_prompts)
            
            # Set config
            await session.set_music_generation_config(
                config=types.LiveMusicGenerationConfig(**config_kwargs)
            )
            
            # Start generation
            await session.play()
            
            # Wait for duration or until we have enough audio
            try:
                await asyncio.wait_for(receive_task, timeout=duration_seconds + 10)
            except asyncio.TimeoutError:
                pass
            
            # Stop playback
            await session.pause()
            
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        raise
    
    if verbose:
        print()  # New line after progress
    
    # Trim to exact duration
    audio_data = bytes(audio_data[:expected_bytes])
    
    # Generate output filename if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create a slug from first prompt
        slug = prompts[0]["text"][:30].lower()
        slug = "".join(c if c.isalnum() else "_" for c in slug).strip("_")
        output_path = f"output/{slug}_{timestamp}.wav"
    
    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write WAV file
    with wave.open(str(output_file), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_data)
    
    # Build metadata
    metadata = {
        "prompts": prompts,
        "duration_seconds": duration_seconds,
        "bpm": bpm,
        "temperature": temperature,
        "seed": seed,
        "density": density,
        "brightness": brightness,
        "scale": scale,
        "top_k": top_k,
        "quality_mode": quality_mode,
        "output_file": str(output_file),
        "generated_at": datetime.now().isoformat(),
        "sample_rate": sample_rate,
        "channels": channels,
        "bit_depth": 16,
    }
    
    # Save metadata JSON
    if save_metadata:
        metadata_file = output_file.with_suffix(".json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        if verbose:
            print(f"📋 Metadata: {metadata_file}")
    
    if verbose:
        print(f"✅ Saved to: {output_file}")
        file_size = output_file.stat().st_size / (1024 * 1024)
        print(f"   Size: {file_size:.2f} MB")
        print(f"🌱 Seed: {seed} (use --seed {seed} to reproduce)")
    
    return output_file, metadata


def main():
    parser = argparse.ArgumentParser(
        description="🎵 Generate music with Google Lyria RealTime",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "dark ambient synths"
  %(prog)s "minimal techno" --duration 60 --bpm 128
  %(prog)s "piano:2.0,ambient:0.5" --scale c_major
  %(prog)s "orchestral score" --quality --output epic.wav
  %(prog)s "deep house" --seed 12345 --duration 120

Prompt syntax:
  "text"           Use weight 1.0
  "text:weight"    Specify weight (e.g., "piano:2.0")
  Multiple prompts can be comma-separated or passed as separate args

Reproducibility:
  Each generation outputs a seed value. Use --seed to reproduce
  the same track at any duration. Metadata is saved to a .json file.

Available scales:
  c_major, c#_major, d_major, d#_major, e_major, f_major,
  f#_major, g_major, g#_major, a_major, a#_major, b_major
        """
    )
    
    parser.add_argument(
        "prompts",
        nargs="+",
        help="Music prompts (e.g., 'minimal techno', 'piano:2.0,drums:0.5')"
    )
    parser.add_argument(
        "-d", "--duration",
        type=int,
        default=30,
        help="Duration in seconds (default: 30)"
    )
    parser.add_argument(
        "--bpm",
        type=int,
        default=120,
        help="Beats per minute, 60-200 (default: 120)"
    )
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=1.0,
        help="Creativity/randomness, 0.0-3.0 (default: 1.0)"
    )
    parser.add_argument(
        "--density",
        type=float,
        help="Note density, 0.0-1.0 (sparse to dense)"
    )
    parser.add_argument(
        "--brightness",
        type=float,
        help="Tonal brightness, 0.0-1.0"
    )
    parser.add_argument(
        "--scale",
        choices=list(SCALES.keys()),
        help="Musical scale/key"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help=f"Random seed for reproducibility ({SEED_MIN}-{SEED_MAX})"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Top-k sampling, 1-1000 (default: 40)"
    )
    parser.add_argument(
        "--quality",
        action="store_true",
        help="Use QUALITY mode (slower but better)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: auto-generated in output/)"
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Don't save metadata JSON file"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show progress information"
    )
    
    args = parser.parse_args()
    
    # Validate ranges
    if not 60 <= args.bpm <= 200:
        parser.error("BPM must be between 60 and 200")
    if not 0.0 <= args.temperature <= 3.0:
        parser.error("Temperature must be between 0.0 and 3.0")
    if args.density is not None and not 0.0 <= args.density <= 1.0:
        parser.error("Density must be between 0.0 and 1.0")
    if args.brightness is not None and not 0.0 <= args.brightness <= 1.0:
        parser.error("Brightness must be between 0.0 and 1.0")
    if args.seed is not None and not SEED_MIN <= args.seed <= SEED_MAX:
        parser.error(f"Seed must be between {SEED_MIN} and {SEED_MAX}")
    if args.top_k is not None and not 1 <= args.top_k <= 1000:
        parser.error("Top-k must be between 1 and 1000")
    
    # Parse prompts
    prompts = parse_prompts(args.prompts)
    if not prompts:
        parser.error("At least one prompt is required")
    
    # Run generation
    try:
        output_file, metadata = asyncio.run(generate_music(
            prompts=prompts,
            duration_seconds=args.duration,
            bpm=args.bpm,
            temperature=args.temperature,
            density=args.density,
            brightness=args.brightness,
            scale=args.scale,
            seed=args.seed,
            top_k=args.top_k,
            quality_mode=args.quality,
            output_path=args.output,
            save_metadata=not args.no_metadata,
            verbose=args.verbose or True,  # Default to verbose for CLI
        ))
        print(f"\n🎶 Done! Your track: {output_file}")
        print(f"🌱 To reproduce: --seed {metadata['seed']}")
    except KeyboardInterrupt:
        print("\n⏹️  Generation cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
