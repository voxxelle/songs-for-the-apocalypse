#!/usr/bin/env python3
"""Submit a track to Apocalypse Radio."""

import argparse
import base64
import json
import sys
from pathlib import Path

import requests


def submit_track(
    api_url: str,
    token: str,
    section_id: str,
    instrument: str,
    audio_path: str,
) -> dict:
    """Submit a track to a section."""
    
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Read and encode audio
    with open(audio_file, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
    
    print(f"📤 Submitting {instrument} track ({audio_file.name})...")
    print(f"   Size: {len(audio_b64) / 1024 / 1024:.2f} MB (base64)")
    
    query = """
    mutation submitTrack($sectionId: String!, $instrument: String!, $audioBase64: String!, $audioFilename: String!) {
        submitTrack(sectionId: $sectionId, instrument: $instrument, audioBase64: $audioBase64, audioFilename: $audioFilename) {
            id
            signedAudioUrl
        }
    }
    """
    
    response = requests.post(
        api_url,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        json={
            "query": query,
            "variables": {
                "sectionId": section_id,
                "instrument": instrument,
                "audioBase64": audio_b64,
                "audioFilename": audio_file.name,
            },
        },
        timeout=120,
    )
    
    result = response.json()
    
    if "errors" in result:
        print(f"❌ Error: {result['errors']}")
    else:
        track = result.get("data", {}).get("submitTrack", {})
        if track:
            print(f"✅ Track submitted! ID: {track.get('id')}")
            if track.get("signedAudioUrl"):
                print(f"   URL: {track['signedAudioUrl'][:80]}...")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Submit a track to Apocalypse Radio")
    parser.add_argument("audio_path", help="Path to WAV/MP3 file")
    parser.add_argument("--token", required=True, help="API auth token")
    parser.add_argument("--section-id", required=True, help="Section ID to submit to")
    parser.add_argument("--instrument", required=True, help="Instrument name (e.g., Drums, Bass, Synths)")
    parser.add_argument("--api", default="https://api.apocalypseradio.xyz/graphql", help="API URL")
    
    args = parser.parse_args()
    
    try:
        result = submit_track(
            api_url=args.api,
            token=args.token,
            section_id=args.section_id,
            instrument=args.instrument,
            audio_path=args.audio_path,
        )
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"❌ Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
