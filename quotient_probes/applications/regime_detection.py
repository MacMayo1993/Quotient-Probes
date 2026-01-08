"""
Regime detection using symmetry coherence as a "lighthouse" signal.

Key insight: Coherence α(t) acts as a probe that lights up when the system
crosses between symmetric and antisymmetric regimes. These crossings ("seams")
often correspond to meaningful transitions in the underlying system.

Applications:
1. EEG: Detect transitions between brain states (rest, active, sleep)
2. Financial: Identify market regime shifts (bull, bear, crisis)
3. Climate: Find phase transitions in weather patterns
4. Audio: Detect scene changes in audio streams

The "lighthouse" metaphor: α(t) sweeps across the critical threshold like
a lighthouse beam, illuminating regime boundaries.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, overload

import numpy as np

from ..core.mdl_decision import critical_coherence
from ..core.symmetry_probe import SymmetryProbe


@dataclass
class RegimeSegment:
    """A contiguous segment with consistent symmetry regime."""

    start_idx: int
    end_idx: int
    regime_type: str  # 'symmetric', 'antisymmetric', or 'mixed'
    mean_coherence: float
    coherence_std: float
    duration: int

    def __repr__(self) -> str:
        return (
            f"Regime({self.regime_type}, "
            f"t=[{self.start_idx}:{self.end_idx}], "
            f"α={self.mean_coherence:.3f}±{self.coherence_std:.3f})"
        )


@dataclass
class SeamPoint:
    """A seam where coherence crosses the critical threshold."""

    index: int
    coherence_before: float
    coherence_after: float
    alpha_crit: float
    direction: str  # 'rising' or 'falling'
    strength: float  # How far from threshold

    def __repr__(self) -> str:
        return (
            f"Seam(t={self.index}, " f"{self.direction}, " f"Δα={self.strength:+.3f})"
        )


class LighthouseDetector:
    """
    Detect regime transitions using rolling coherence analysis.

    The detector:
    1. Computes rolling coherence α(t) over sliding windows
    2. Identifies seams where α(t) crosses α_crit
    3. Segments the time series into coherent regimes

    Metaphor: α(t) is a lighthouse beam that illuminates transitions
    between symmetric and antisymmetric regimes.

    Usage:
        >>> detector = LighthouseDetector(window_size=64)
        >>> seams, regimes = detector.detect(time_series)
        >>> print(f"Found {len(seams)} regime transitions")

    Attributes:
        window_size: Size of rolling window
        overlap: Overlap ratio between windows
        involution: Symmetry operator
        K_lift: Orientation cost
    """

    def __init__(
        self,
        window_size: int = 64,
        overlap: float = 0.5,
        involution: str = "reverse",
        K_lift: float = 1.0,
        min_regime_duration: int = 3,
    ):
        """
        Initialize lighthouse detector.

        Args:
            window_size: Window size for rolling coherence
            overlap: Overlap ratio (0.0 to 1.0)
            involution: Involution operator ('reverse' for time series)
            K_lift: Orientation cost
            min_regime_duration: Minimum duration of a regime (in windows)

        Example:
            >>> detector = LighthouseDetector(window_size=128, overlap=0.75)
        """
        if not 0 <= overlap < 1:
            raise ValueError(f"overlap must be in [0, 1), got {overlap}")

        self.window_size = window_size
        self.overlap = overlap
        self.step = int(window_size * (1 - overlap))
        self.involution = involution
        self.K_lift = K_lift
        self.min_regime_duration = min_regime_duration

        # Compute critical threshold
        self.alpha_crit = critical_coherence(window_size, K_lift)

    def compute_rolling_coherence(
        self,
        time_series: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute coherence over rolling windows.

        Args:
            time_series: Input time series (1D array)

        Returns:
            Tuple of (window_centers, coherences)

        Example:
            >>> centers, alphas = detector.compute_rolling_coherence(data)
            >>> plt.plot(centers, alphas)
        """
        n = len(time_series)
        window_centers = []
        coherences = []

        for start in range(0, n - self.window_size + 1, self.step):
            end = start + self.window_size
            window = time_series[start:end]

            # Compute coherence for this window
            probe = SymmetryProbe(
                window, involution=self.involution, K_lift=self.K_lift
            )
            alpha = probe.get_coherence()

            center = (start + end) // 2
            window_centers.append(center)
            coherences.append(alpha)

        return np.array(window_centers), np.array(coherences)

    def detect_seams(
        self,
        coherences: np.ndarray,
        window_centers: np.ndarray,
    ) -> List[SeamPoint]:
        """
        Detect seams where coherence crosses the critical threshold.

        Args:
            coherences: Coherence time series
            window_centers: Center indices of windows

        Returns:
            List of SeamPoint objects

        Example:
            >>> centers, alphas = detector.compute_rolling_coherence(data)
            >>> seams = detector.detect_seams(alphas, centers)
        """
        seams = []

        for i in range(len(coherences) - 1):
            alpha_before = coherences[i]
            alpha_after = coherences[i + 1]

            # Check for crossing
            crosses_up = (alpha_before < self.alpha_crit) and (
                alpha_after >= self.alpha_crit
            )
            crosses_down = (alpha_before >= self.alpha_crit) and (
                alpha_after < self.alpha_crit
            )

            if crosses_up or crosses_down:
                direction = "rising" if crosses_up else "falling"

                # Strength: how far the coherence moved relative to threshold
                strength = alpha_after - alpha_before

                seam = SeamPoint(
                    index=window_centers[i + 1],
                    coherence_before=alpha_before,
                    coherence_after=alpha_after,
                    alpha_crit=self.alpha_crit,
                    direction=direction,
                    strength=strength,
                )
                seams.append(seam)

        return seams

    def segment_regimes(
        self,
        coherences: np.ndarray,
        window_centers: np.ndarray,
    ) -> List[RegimeSegment]:
        """
        Segment time series into coherent regimes.

        Args:
            coherences: Coherence time series
            window_centers: Center indices of windows

        Returns:
            List of RegimeSegment objects

        Example:
            >>> centers, alphas = detector.compute_rolling_coherence(data)
            >>> regimes = detector.segment_regimes(alphas, centers)
        """
        if len(coherences) == 0:
            return []

        regimes = []
        current_regime_start = 0
        current_regime_type = self._classify_coherence(coherences[0])

        for i in range(1, len(coherences)):
            regime_type = self._classify_coherence(coherences[i])

            # Check if regime changed
            if regime_type != current_regime_type or i == len(coherences) - 1:
                # End current regime
                end_idx = i if regime_type != current_regime_type else i + 1

                # Check minimum duration
                if end_idx - current_regime_start >= self.min_regime_duration:
                    segment_coherences = coherences[current_regime_start:end_idx]

                    regime = RegimeSegment(
                        start_idx=window_centers[current_regime_start],
                        end_idx=window_centers[end_idx - 1],
                        regime_type=current_regime_type,
                        mean_coherence=np.mean(segment_coherences),
                        coherence_std=np.std(segment_coherences),
                        duration=end_idx - current_regime_start,
                    )
                    regimes.append(regime)

                # Start new regime
                current_regime_start = i
                current_regime_type = regime_type

        return regimes

    def _classify_coherence(self, alpha: float) -> str:
        """Classify coherence value into regime type."""
        margin = 0.05  # Small margin around threshold

        if alpha > self.alpha_crit + margin:
            return "symmetric"
        elif alpha < self.alpha_crit - margin:
            return "antisymmetric"
        else:
            return "mixed"

    @overload
    def detect(
        self,
        time_series: np.ndarray,
        return_coherence: Literal[False] = False,
    ) -> Tuple[List[SeamPoint], List[RegimeSegment]]: ...

    @overload
    def detect(
        self,
        time_series: np.ndarray,
        return_coherence: Literal[True] = True,
    ) -> Tuple[List[SeamPoint], List[RegimeSegment], np.ndarray, np.ndarray]: ...

    def detect(
        self,
        time_series: np.ndarray,
        return_coherence: bool = False,
    ) -> (
        Tuple[List[SeamPoint], List[RegimeSegment]]
        | Tuple[List[SeamPoint], List[RegimeSegment], np.ndarray, np.ndarray]
    ):
        """
        Detect regime transitions in time series.

        Args:
            time_series: Input time series (1D array)
            return_coherence: If True, also return (window_centers, coherences)

        Returns:
            Tuple of (seams, regimes) or (seams, regimes, centers, coherences)

        Example:
            >>> seams, regimes = detector.detect(eeg_signal)
            >>> print(f"Found {len(seams)} transitions across {len(regimes)} regimes")
        """
        # Compute rolling coherence
        window_centers, coherences = self.compute_rolling_coherence(time_series)

        # Detect seams
        seams = self.detect_seams(coherences, window_centers)

        # Segment regimes
        regimes = self.segment_regimes(coherences, window_centers)

        if return_coherence:
            return seams, regimes, window_centers, coherences
        else:
            return seams, regimes

    def plot_detection(
        self,
        time_series: np.ndarray,
        seams: List[SeamPoint],
        regimes: List[RegimeSegment],
        window_centers: np.ndarray,
        coherences: np.ndarray,
        figsize: Tuple[int, int] = (14, 8),
    ):
        """
        Visualize regime detection results.

        Args:
            time_series: Original time series
            seams: Detected seam points
            regimes: Detected regimes
            window_centers: Window center indices
            coherences: Coherence time series
            figsize: Figure size

        Example:
            >>> seams, regimes, centers, alphas = detector.detect(data, return_coherence=True)
            >>> detector.plot_detection(data, seams, regimes, centers, alphas)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not available for plotting")
            return

        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Plot 1: Original time series with regime shading
        ax = axes[0]
        t = np.arange(len(time_series))
        ax.plot(t, time_series, "k-", linewidth=1, alpha=0.7, label="Time series")

        # Shade regimes
        colors = {"symmetric": "blue", "antisymmetric": "red", "mixed": "gray"}
        for regime in regimes:
            ax.axvspan(
                regime.start_idx,
                regime.end_idx,
                alpha=0.2,
                color=colors.get(regime.regime_type, "gray"),
                label=f"{regime.regime_type}" if regime == regimes[0] else None,
            )

        # Mark seams
        for seam in seams:
            color = "green" if seam.direction == "rising" else "orange"
            ax.axvline(
                seam.index, color=color, linestyle="--", linewidth=1.5, alpha=0.7
            )

        ax.set_ylabel("Amplitude", fontsize=11)
        ax.set_title(
            "Time Series with Detected Regimes", fontsize=13, fontweight="bold"
        )
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Plot 2: Coherence timeline with seams
        ax = axes[1]
        ax.plot(
            window_centers,
            coherences,
            "b-",
            linewidth=2,
            marker="o",
            markersize=4,
            label="Coherence α(t)",
        )

        # Critical threshold
        ax.axhline(
            self.alpha_crit,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"α_crit = {self.alpha_crit:.4f}",
        )

        # Shade decision regions
        ax.axhspan(
            self.alpha_crit, 1.0, alpha=0.1, color="blue", label="Symmetric regime"
        )
        ax.axhspan(
            0, self.alpha_crit, alpha=0.1, color="red", label="Antisymmetric regime"
        )

        # Mark seams
        for seam in seams:
            color = "green" if seam.direction == "rising" else "orange"
            ax.plot(
                seam.index,
                seam.coherence_after,
                "o",
                color=color,
                markersize=10,
                markeredgewidth=2,
                markeredgecolor="black",
            )

        ax.set_xlabel("Time", fontsize=11)
        ax.set_ylabel("Coherence α", fontsize=11)
        ax.set_title(
            f"Lighthouse Signal (window={self.window_size}, overlap={self.overlap})",
            fontsize=13,
            fontweight="bold",
        )
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

        plt.tight_layout()
        return fig

    def get_statistics(
        self, seams: List[SeamPoint], regimes: List[RegimeSegment]
    ) -> Dict[str, Any]:
        """
        Compute summary statistics.

        Args:
            seams: Detected seams
            regimes: Detected regimes

        Returns:
            Dictionary with statistics
        """
        if len(regimes) == 0:
            return {
                "num_seams": len(seams),
                "num_regimes": 0,
                "avg_regime_duration": 0,
            }

        # Regime type distribution
        regime_counts: Dict[str, int] = {}
        for regime in regimes:
            regime_counts[regime.regime_type] = (
                regime_counts.get(regime.regime_type, 0) + 1
            )

        # Seam direction distribution
        rising_seams = sum(1 for s in seams if s.direction == "rising")
        falling_seams = sum(1 for s in seams if s.direction == "falling")

        # Duration statistics
        durations = [r.duration for r in regimes]

        return {
            "num_seams": len(seams),
            "num_regimes": len(regimes),
            "regime_type_counts": regime_counts,
            "rising_seams": rising_seams,
            "falling_seams": falling_seams,
            "avg_regime_duration": np.mean(durations),
            "std_regime_duration": np.std(durations),
            "min_regime_duration": np.min(durations),
            "max_regime_duration": np.max(durations),
            "avg_coherence_symmetric": (
                np.mean(
                    [r.mean_coherence for r in regimes if r.regime_type == "symmetric"]
                )
                if any(r.regime_type == "symmetric" for r in regimes)
                else None
            ),
            "avg_coherence_antisymmetric": (
                np.mean(
                    [
                        r.mean_coherence
                        for r in regimes
                        if r.regime_type == "antisymmetric"
                    ]
                )
                if any(r.regime_type == "antisymmetric" for r in regimes)
                else None
            ),
        }


def generate_synthetic_regime_series(
    regime_durations: List[int] = [100, 150, 200, 100],
    regime_alphas: List[float] = [0.7, 0.3, 0.8, 0.2],
    window_size: int = 64,
    noise_level: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate synthetic time series with known regime structure.

    Args:
        regime_durations: Duration of each regime (in samples)
        regime_alphas: Target coherence for each regime
        window_size: Window size for regime construction
        noise_level: Noise level
        seed: Random seed

    Returns:
        Synthetic time series with regime transitions

    Example:
        >>> ts = generate_synthetic_regime_series([200, 300], [0.8, 0.2])
        >>> detector = LighthouseDetector()
        >>> seams, regimes = detector.detect(ts)
    """
    np.random.seed(seed)

    segments = []

    for duration, target_alpha in zip(regime_durations, regime_alphas):
        # Generate segment with target coherence
        # For reverse involution, palindromic = symmetric, anti-palindromic = antisymmetric

        if target_alpha > 0.5:
            # Symmetric regime: create palindromic-ish signal
            half = duration // 2
            first_half = np.random.randn(half)
            second_half = first_half[::-1] + np.random.randn(half) * noise_level
            segment = np.concatenate([first_half, second_half])
        else:
            # Antisymmetric regime: create anti-palindromic signal
            half = duration // 2
            first_half = np.random.randn(half)
            second_half = -first_half[::-1] + np.random.randn(half) * noise_level
            segment = np.concatenate([first_half, second_half])

        # Pad or trim to exact duration
        if len(segment) < duration:
            segment = np.pad(segment, (0, duration - len(segment)), mode="edge")
        else:
            segment = segment[:duration]

        segments.append(segment)

    return np.concatenate(segments)
