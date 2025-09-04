
"""
Standalone test script for Veo 3 video generation using npcpy.llm_funcs
"""

import os
from pathlib import Path
from npcpy.llm_funcs import gen_video

def test_veo3_basic():
    """Test basic Veo 3 video generation"""
    print("=== Testing Basic Veo 3 Video Generation ===")
    
    prompt = "a close-up shot of a golden retriever playing in a field of sunflowers"
    
    result = gen_video(
        prompt=prompt,
        provider="gemini",
        model="veo-3.0-generate-preview",
        output_path="",
    )
    
    print(f"✅ Success: {result['output']}")
    return result

def test_veo3_with_negative_prompt():
    """Test Veo 3 with negative prompt"""
    print("\n=== Testing Veo 3 with Negative Prompt ===")
    
    prompt = "a fluffy cat sitting on a windowsill watching birds outside"
    negative_prompt = "barking, dogs, noise"
    
    result = gen_video(
        prompt=prompt,
        provider="gemini",
        model="veo-3.0-generate-preview",
        negative_prompt=negative_prompt,
        output_path="",
    )
    
    print(f"✅ Success with negative prompt: {result['output']}")
    return result

def test_veo3_custom_output():
    """Test Veo 3 with custom output path"""
    print("\n=== Testing Veo 3 with Custom Output Path ===")
    
    prompt = "a time-lapse of a flower blooming in spring sunlight"
    custom_output = os.path.expanduser("~/Desktop/my_veo3_test.mp4")
    
    result = gen_video(
        prompt=prompt,
        provider="gemini",
        model="veo-3.0-generate-preview",
        output_path=custom_output,
    )
    
    print(f"✅ Success with custom path: {result['output']}")
    
    if os.path.exists(custom_output):
        file_size = os.path.getsize(custom_output)
        print(f"📁 File created: {custom_output} ({file_size} bytes)")
    else:
        print("⚠️  File not found at expected location")
        
    return result

def main():
    """Main test function"""
    print("🎬 Veo 3 Video Generation Test Script")
    print("=" * 50)
    
    output_dir = Path.home() / ".npcsh" / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    
    test_veo3_basic()
    
    
    test_veo3_with_negative_prompt()
    
    
    test_veo3_custom_output()
    
    
    video_files = list(output_dir.glob("*.mp4"))
    if video_files:
        print(f"\n🎬 Generated videos in {output_dir}:")
        for video_file in video_files:
            size = video_file.stat().st_size
            print(f"  - {video_file.name} ({size} bytes)")
    else:
        print(f"\n📂 No video files found in {output_dir}")

if __name__ == "__main__":
    main()