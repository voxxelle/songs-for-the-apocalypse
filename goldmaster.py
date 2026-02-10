#!/usr/bin/env python3
"""
🎨 Gold Master Image Generator
Generate album art and cover images using Google's Gemini 2.5 Flash Image model.

Usage:
    python goldmaster.py "dark ambient album cover, post-apocalyptic cityscape"
    python goldmaster.py "vinyl record cover, synthwave aesthetic" --output cover.png
    python goldmaster.py "add flames to the background" --input existing.png

Environment:
    GOOGLE_API_KEY or GEMINI_API_KEY must be set.
"""

import argparse
import base64
import json
import os
import sys
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

try:
    from PIL import Image
    import io
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def get_api_key() -> str:
    """Get API key from environment."""
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        print("Error: No API key found.")
        print("Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
        sys.exit(1)
    return key


# Available image generation models
IMAGE_MODELS = {
    "flash": "gemini-2.0-flash-exp-image-generation",
    "imagen": "imagen-3.0-generate-002",
}
DEFAULT_MODEL = "flash"


def generate_image(
    prompt: str,
    input_image: str | None = None,
    output_path: str | None = None,
    model: str | None = None,
    save_metadata: bool = True,
    verbose: bool = False,
) -> tuple[Path, dict]:
    """
    Generate an image using Gemini 2.5 Flash Image.
    
    Args:
        prompt: Text description of the image to generate
        input_image: Optional path to input image for editing
        output_path: Output file path (default: auto-generated)
        model: Model to use (flash, imagen, or full model name)
        save_metadata: Save generation params to JSON file
        verbose: Print progress information
        
    Returns:
        Tuple of (Path to image file, metadata dict)
    """
    api_key = get_api_key()
    
    # Resolve model name
    if model is None:
        model = DEFAULT_MODEL
    model_name = IMAGE_MODELS.get(model, model)
    
    # Initialize client
    client = genai.Client(api_key=api_key)
    
    if verbose:
        print(f"🎨 Generating image...")
        print(f"   Model: {model_name}")
        print(f"   Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        if input_image:
            print(f"   Input: {input_image}")
    
    # Build content
    contents = []
    
    # Add input image if provided
    if input_image:
        input_path = Path(input_image)
        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {input_image}")
        
        if HAS_PIL:
            # Use PIL to load and pass image
            img = Image.open(input_path)
            contents.append(prompt)
            contents.append(img)
        else:
            # Fallback: base64 encode
            with open(input_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            
            # Determine MIME type
            suffix = input_path.suffix.lower()
            mime_types = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }
            mime_type = mime_types.get(suffix, "image/png")
            
            contents.append(types.Content(
                parts=[
                    types.Part(text=prompt),
                    types.Part(inline_data=types.Blob(
                        mime_type=mime_type,
                        data=img_data,
                    )),
                ]
            ))
    else:
        contents.append(prompt)
    
    # Generate image
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
            ),
        )
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        raise
    
    # Extract image from response
    image_data = None
    text_response = None
    
    for part in response.candidates[0].content.parts:
        if hasattr(part, 'text') and part.text:
            text_response = part.text
            if verbose:
                print(f"   Model says: {text_response[:100]}...")
        elif hasattr(part, 'inline_data') and part.inline_data:
            image_data = part.inline_data.data
            mime_type = part.inline_data.mime_type
    
    if image_data is None:
        raise ValueError("No image was generated. The model may have refused the prompt.")
    
    # Determine output format
    if mime_type == "image/png":
        ext = ".png"
    elif mime_type in ("image/jpeg", "image/jpg"):
        ext = ".jpg"
    elif mime_type == "image/webp":
        ext = ".webp"
    else:
        ext = ".png"
    
    # Generate output filename if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create a slug from prompt
        slug = prompt[:40].lower()
        slug = "".join(c if c.isalnum() else "_" for c in slug).strip("_")
        output_path = f"output/{slug}_{timestamp}{ext}"
    
    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Decode and save image
    if isinstance(image_data, str):
        image_bytes = base64.b64decode(image_data)
    else:
        image_bytes = image_data
    
    with open(output_file, "wb") as f:
        f.write(image_bytes)
    
    # Build metadata
    metadata = {
        "prompt": prompt,
        "input_image": input_image,
        "output_file": str(output_file),
        "mime_type": mime_type,
        "text_response": text_response,
        "generated_at": datetime.now().isoformat(),
        "model": model_name,
    }
    
    # Save metadata JSON
    if save_metadata:
        metadata_file = output_file.with_suffix(".json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        if verbose:
            print(f"📋 Metadata: {metadata_file}")
    
    if verbose:
        file_size = output_file.stat().st_size / 1024
        print(f"✅ Saved to: {output_file}")
        print(f"   Size: {file_size:.1f} KB")
    
    return output_file, metadata


def main():
    parser = argparse.ArgumentParser(
        description="🎨 Generate album art with Gemini 2.5 Flash Image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "dark ambient album cover, post-apocalyptic cityscape at night"
  %(prog)s "vinyl record cover, synthwave aesthetic, neon colors" --output cover.png
  %(prog)s "add fire effects to the sky" --input original.png
  %(prog)s "gold master vinyl pressing, apocalyptic theme" -v

Prompt tips for album art:
  - Include style: "album cover", "vinyl record", "CD artwork"
  - Add genre vibes: "dark ambient", "synthwave", "industrial"
  - Describe scene: "abandoned city", "cosmic void", "neon wasteland"
  - Specify mood: "haunting", "epic", "melancholic", "ominous"
        """
    )
    
    parser.add_argument(
        "prompt",
        help="Text description of the image to generate"
    )
    parser.add_argument(
        "-i", "--input",
        help="Input image for editing (optional)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: auto-generated in output/)"
    )
    parser.add_argument(
        "-m", "--model",
        default="flash",
        help="Model to use: flash, imagen, or full model name (default: flash)"
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
    
    # Run generation
    try:
        output_file, metadata = generate_image(
            prompt=args.prompt,
            input_image=args.input,
            output_path=args.output,
            model=args.model,
            save_metadata=not args.no_metadata,
            verbose=args.verbose or True,  # Default to verbose for CLI
        )
        print(f"\n🖼️  Done! Your image: {output_file}")
    except KeyboardInterrupt:
        print("\n⏹️  Generation cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
