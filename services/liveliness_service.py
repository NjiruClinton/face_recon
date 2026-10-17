import cv2
import numpy as np
from typing import Dict, Tuple


class LivelinessService:
    """
    Silent Face Anti-Spoofing Service

    This service implements multiple techniques to detect if a face is live/real
    without requiring user interaction (silent detection):

    1. Texture Analysis - Real faces have natural skin texture patterns
    2. Color Distribution - Real faces have specific color distributions
    3. Edge Detection - Photos/screens have different edge characteristics
    4. Frequency Analysis - Real faces have different frequency patterns than screens
    5. Reflection Detection - Screens often have reflective properties
    """

    def __init__(self):
        # Initialize Laplacian kernel for edge detection
        self.laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

        # Thresholds for different tests (can be fine-tuned)
        self.thresholds = {
            'texture_variance': 100,  # Minimum texture variance for real faces
            'edge_density': 0.1,      # Minimum edge density
            'color_diversity': 20,    # Minimum color channel diversity
            'frequency_ratio': 0.3,   # High/low frequency ratio threshold
        }

    def analyze_liveliness(self, image: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Dict:
        """
        Analyze if the detected face appears to be live/real

        Args:
            image: Input image (BGR format)
            face_bbox: Face bounding box (x, y, width, height)

        Returns:
            Dict with liveliness analysis results
        """
        x, y, w, h = face_bbox
        face_region = image[y:y+h, x:x+w]

        if face_region.size == 0:
            return {
                'is_live': False,
                'confidence': 0.0,
                'reason': 'Invalid face region',
                'scores': {}
            }

        # Run all anti-spoofing tests
        scores = {}
        scores['texture'] = self._analyze_texture(face_region)
        scores['edges'] = self._analyze_edges(face_region)
        scores['color'] = self._analyze_color_distribution(face_region)
        scores['frequency'] = self._analyze_frequency_domain(face_region)
        scores['reflection'] = self._detect_screen_reflection(face_region)

        # Calculate overall liveliness score
        overall_score, is_live, reason = self._calculate_final_score(scores)

        return {
            'is_live': is_live,
            'confidence': overall_score,
            'reason': reason,
            'scores': scores
        }

    def _analyze_texture(self, face_region: np.ndarray) -> float:
        """
        Analyze skin texture patterns. Real faces have natural texture variance.
        Photos/screens tend to have smoother, more uniform textures.
        """
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        # Calculate local variance using a sliding window
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)

        # Calculate local means and variances
        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        sqr_diff = (gray.astype(np.float32) - mean) ** 2
        local_var = cv2.filter2D(sqr_diff, -1, kernel)

        # Average texture variance across the face
        avg_variance = np.mean(local_var)

        # Normalize score (higher variance = more likely to be real)
        texture_score = min(avg_variance / self.thresholds['texture_variance'], 1.0)

        return float(texture_score)

    def _analyze_edges(self, face_region: np.ndarray) -> float:
        """
        Analyze edge characteristics. Real faces have softer, more natural edges.
        Photos/screens often have sharper, more defined edges.
        """
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        # Apply Laplacian edge detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        edge_variance = np.var(laplacian)

        # Calculate edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Real faces should have moderate edge density (not too sharp, not too smooth)
        if edge_density < self.thresholds['edge_density']:
            edge_score = edge_density / self.thresholds['edge_density']
        else:
            # Penalize overly sharp edges (typical of photos/screens)
            edge_score = max(0.0, 1.0 - (edge_density - self.thresholds['edge_density']) * 2)

        return float(edge_score)

    def _analyze_color_distribution(self, face_region: np.ndarray) -> float:
        """
        Analyze color distribution patterns. Real faces have natural color variations.
        Screens/photos may have artificial color casts or limited color ranges.
        """
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)

        # Analyze color diversity in each channel
        color_diversity = 0

        # HSV analysis (focus on saturation and value)
        sat_std = np.std(hsv[:, :, 1])  # Saturation standard deviation
        val_std = np.std(hsv[:, :, 2])  # Value standard deviation

        # LAB analysis (focus on a and b channels for color information)
        a_std = np.std(lab[:, :, 1])
        b_std = np.std(lab[:, :, 2])

        # Combine color diversity metrics
        color_diversity = (sat_std + val_std + a_std + b_std) / 4

        # Normalize score
        color_score = min(color_diversity / self.thresholds['color_diversity'], 1.0)

        return float(color_score)

    def _analyze_frequency_domain(self, face_region: np.ndarray) -> float:
        """
        Analyze frequency domain characteristics. Screens tend to have different
        frequency patterns than natural faces due to pixel structure.
        """
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        # Apply FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)

        # Divide frequency domain into high and low frequency regions
        rows, cols = magnitude_spectrum.shape
        crow, ccol = rows // 2, cols // 2

        # Create masks for low and high frequency regions
        low_freq_mask = np.zeros((rows, cols), np.uint8)
        low_freq_mask[crow-30:crow+30, ccol-30:ccol+30] = 1

        high_freq_mask = 1 - low_freq_mask

        # Calculate energy in low and high frequency regions
        low_freq_energy = np.sum(magnitude_spectrum * low_freq_mask)
        high_freq_energy = np.sum(magnitude_spectrum * high_freq_mask)

        # Calculate ratio (real faces should have balanced frequency content)
        if low_freq_energy > 0:
            freq_ratio = high_freq_energy / low_freq_energy
            # Normalize the ratio
            freq_score = min(freq_ratio / self.thresholds['frequency_ratio'], 1.0)
        else:
            freq_score = 0.0

        return float(freq_score)

    def _detect_screen_reflection(self, face_region: np.ndarray) -> float:
        """
        Detect screen-like reflections or uniform illumination patterns
        that are common in spoofed faces from screens or photos.
        """
        # Convert to HSV for better illumination analysis
        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)

        # Analyze the Value (brightness) channel
        brightness = hsv[:, :, 2]

        # Look for uniform brightness patterns (typical of screens)
        brightness_variance = np.var(brightness)

        # Calculate gradient uniformity
        grad_x = cv2.Sobel(brightness, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(brightness, cv2.CV_64F, 0, 1, ksize=3)

        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_uniformity = np.var(gradient_magnitude)

        # Check for periodic patterns (screen pixels)
        # Apply morphological operations to detect regular patterns
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(brightness, cv2.MORPH_OPEN, kernel)
        pattern_score = np.mean(np.abs(brightness - opening))

        # Combine metrics (higher values indicate more natural, non-screen-like properties)
        reflection_score = min((brightness_variance + gradient_uniformity) / 10000, 1.0)

        # Penalize regular patterns
        if pattern_score > 5:
            reflection_score *= 0.5

        return float(reflection_score)

    def _calculate_final_score(self, scores: Dict[str, float]) -> Tuple[float, bool, str]:
        """
        Calculate the final liveliness score and determination.

        Returns:
            Tuple of (overall_score, is_live, reason)
        """
        # Weight different tests based on their reliability
        weights = {
            'texture': 0.25,
            'edges': 0.20,
            'color': 0.20,
            'frequency': 0.20,
            'reflection': 0.15
        }

        # Calculate weighted average
        weighted_score = sum(scores[test] * weights[test] for test in weights.keys())

        # Determine if face is live (threshold of 0.6)
        is_live = weighted_score >= 0.6

        # Generate reason based on failing tests
        if not is_live:
            failing_tests = [test for test, score in scores.items() if score < 0.5]
            if failing_tests:
                reason = f"Failed anti-spoofing tests: {', '.join(failing_tests)}"
            else:
                reason = "Overall confidence too low for live face detection"
        else:
            reason = "Face appears to be live based on anti-spoofing analysis"

        return weighted_score, is_live, reason
