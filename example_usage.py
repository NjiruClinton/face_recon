#!/usr/bin/env python3
"""
Example script demonstrating how to use the multi-frame liveliness detection endpoint.

This script shows how to:
1. Send single frame requests
2. Send multi-frame requests
3. Process and analyze the detailed scores returned for each frame
"""

import requests
import json
from pathlib import Path


def test_single_frame_liveliness(server_url: str, image_path: str):
    """Test liveliness detection with a single frame."""

    print(f"\n=== Testing Single Frame Liveliness ===")
    print(f"Image: {image_path}")

    url = f"{server_url}/liveliness"

    try:
        with open(image_path, 'rb') as f:
            files = {'files': (Path(image_path).name, f, 'image/jpeg')}
            response = requests.post(url, files=files)

        if response.status_code == 200:
            result = response.json()
            print(f"Success!")
            print(f"Overall Live: {result['overall_is_live']}")
            print(f"Confidence: {result['overall_confidence']:.3f}")
            print(f"Reason: {result['reason']}")

            # Display detailed scores for the single frame
            if result['frames'] and result['frames'][0]['detailed_scores']:
                scores = result['frames'][0]['detailed_scores']
                print(f"\nDetailed Scores:")
                for test_name, score in scores.items():
                    print(f"  {test_name.capitalize()}: {score:.3f}")

        else:
            print(f"Error: {response.status_code}")
            print(response.text)

    except FileNotFoundError:
        print(f"File not found: {image_path}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")


def test_multi_frame_liveliness(server_url: str, image_paths: list):
    """Test liveliness detection with multiple frames."""

    print(f"\n=== Testing Multi-Frame Liveliness ===")
    print(f"Frames: {len(image_paths)}")

    url = f"{server_url}/liveliness"

    try:
        files = []
        for i, image_path in enumerate(image_paths):
            try:
                f = open(image_path, 'rb')
                files.append(('files', (f"frame_{i}.jpg", f, 'image/jpeg')))
            except FileNotFoundError:
                print(f"Skipping missing file: {image_path}")
                continue

        if not files:
            print("No valid files to process")
            return

        response = requests.post(url, files=files)

        # Close all file handles
        for _, (_, f, _) in files:
            f.close()

        if response.status_code == 200:
            result = response.json()
            print(f"Success!")
            print(f"Overall Live: {result['overall_is_live']}")
            print(f"Overall Confidence: {result['overall_confidence']:.3f}")
            print(f"Valid Frames: {result['valid_frames']}/{result['frame_count']}")

            if 'consistency_ratio' in result:
                print(f"Consistency Ratio: {result['consistency_ratio']:.1%}")

            print(f"Reason: {result['reason']}")

            # Display aggregated scores
            if result['aggregated_scores']:
                print(f"\nAggregated Scores (median across frames):")
                for test_name, score in result['aggregated_scores'].items():
                    print(f"  {test_name.capitalize()}: {score:.3f}")

            # Display per-frame results
            print(f"\nPer-Frame Results:")
            for frame in result['frames']:
                status = "W" if frame['is_live'] else "L"
                face_detected = "👤" if frame['face_detected'] else "??"
                print(f"  Frame {frame['frame_index']}: {status} {face_detected} "
                      f"Confidence: {frame['confidence']:.3f} - {frame['reason']}")

        else:
            print(f" Error: {response.status_code}")
            print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"E Request failed: {e}")


def analyze_frame_scores(frames_data):
    """Analyze and compare scores across multiple frames."""

    print(f"\n=== Frame Score Analysis ===")

    valid_frames = [f for f in frames_data if f['face_detected'] and f['detailed_scores']]

    if not valid_frames:
        print("No valid frames to analyze")
        return

    # Calculate score statistics across frames
    score_types = ['texture', 'edges', 'color', 'frequency', 'reflection']

    print(f"Score analysis across {len(valid_frames)} valid frames:\n")
    print(f"{'Test':<12} {'Min':<6} {'Max':<6} {'Avg':<6} {'Std':<6}")
    print("-" * 38)

    for score_type in score_types:
        scores = [frame['detailed_scores'].get(score_type, 0) for frame in valid_frames]
        if scores:
            min_score = min(scores)
            max_score = max(scores)
            avg_score = sum(scores) / len(scores)
            std_score = (sum((x - avg_score) ** 2 for x in scores) / len(scores)) ** 0.5

            print(f"{score_type.capitalize():<12} {min_score:<6.3f} {max_score:<6.3f} {avg_score:<6.3f} {std_score:<6.3f}")

    # Identify most/least consistent tests
    print(f"\nConsistency Analysis:")
    consistencies = {}
    for score_type in score_types:
        scores = [frame['detailed_scores'].get(score_type, 0) for frame in valid_frames]
        if len(scores) > 1:
            avg_score = sum(scores) / len(scores)
            std_score = (sum((x - avg_score) ** 2 for x in scores) / len(scores)) ** 0.5
            consistencies[score_type] = std_score

    if consistencies:
        most_consistent = min(consistencies.items(), key=lambda x: x[1])
        least_consistent = max(consistencies.items(), key=lambda x: x[1])

        print(f"Most consistent test: {most_consistent[0]} (std: {most_consistent[1]:.3f})")
        print(f"Least consistent test: {least_consistent[0]} (std: {least_consistent[1]:.3f})")


def main():
    """Main example function."""

    # Configuration
    SERVER_URL = "http://127.0.0.1:8000"

    # Example paths - replace with your actual image paths
    SINGLE_IMAGE = "/path/to/your/test_image.jpg"
    MULTIPLE_IMAGES = [
        "/path/to/frame1.jpg",
        "/path/to/frame2.jpg",
        "/path/to/frame3.jpg"
    ]

    print("Face Liveliness Detection - Multi-Frame Example")
    print("=" * 50)

    # Test single frame
    print("\n1. Testing single frame detection...")
    test_single_frame_liveliness(SERVER_URL, SINGLE_IMAGE)

    # Test multiple frames
    print("\n2. Testing multi-frame detection...")
    response = test_multi_frame_liveliness(SERVER_URL, MULTIPLE_IMAGES)

    print("\n3. Advanced analysis example...")
    print("(This would typically use the frames data from a real response)")

    # Example of how to analyze results programmatically
    example_frames = [
        {"frame_index": 0, "face_detected": True, "is_live": True, "confidence": 0.78,
         "detailed_scores": {"texture": 0.85, "edges": 0.72, "color": 0.68, "frequency": 0.81, "reflection": 0.75}},
        {"frame_index": 1, "face_detected": True, "is_live": True, "confidence": 0.82,
         "detailed_scores": {"texture": 0.88, "edges": 0.75, "color": 0.71, "frequency": 0.79, "reflection": 0.78}},
        {"frame_index": 2, "face_detected": True, "is_live": True, "confidence": 0.76,
         "detailed_scores": {"texture": 0.81, "edges": 0.70, "color": 0.65, "frequency": 0.83, "reflection": 0.72}}
    ]

    analyze_frame_scores(example_frames)

    print(f"\n=== Usage Tips ===")
    print("• Use multiple frames (3-5) for more reliable detection")
    print("• Frames should be from the same person in the same session")
    print("• Check consistency_ratio - values >0.8 indicate reliable results")
    print("• Individual test scores help identify specific spoofing methods")
    print("• Low texture scores often indicate printed photos")
    print("• Low frequency scores may indicate screen display attacks")


if __name__ == "__main__":
    main()
