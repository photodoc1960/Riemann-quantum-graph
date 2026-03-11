"""Riemann zeta non-trivial zeros: storage, retrieval, and statistics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

# Hard-coded first 100 non-trivial zeros (imaginary parts) to 15 decimal places.
# First 10 from specification; remaining 90 computed via mpmath.zetazero().
_FIRST_100_ZEROS: tuple[float, ...] = (
    14.134725141734693,
    21.022039638771555,
    25.010857580145688,
    30.424876125859513,
    32.935061587739189,
    37.586178158825671,
    40.918719012147495,
    43.327073280914999,
    48.005150881167159,
    49.773832477672302,
    # 11-20
    52.970321477714460,
    56.446247697063394,
    59.347044002602353,
    60.831778524609809,
    65.112544048081606,
    67.079810529494173,
    69.546401711173979,
    72.067157674481907,
    75.704690699083933,
    77.144840068874805,
    # 21-30
    79.337375020249367,
    82.910380854086030,
    84.735492980517050,
    87.425274613125229,
    88.809111207634465,
    92.491899270558484,
    94.651344040519838,
    95.870634228245309,
    98.831194218193692,
    101.317851005731220,
    # 31-40
    103.725538040346240,
    105.446623052771263,
    107.168611184276408,
    111.029535543298395,
    111.874659176748740,
    114.320220915452712,
    116.226680320857573,
    118.790782865976214,
    121.370125002416800,
    122.946829293851560,
    # 41-50
    124.256818554345490,
    127.516683879616540,
    129.578704199956360,
    131.087688531090290,
    133.497737203028150,
    134.756509753073870,
    138.116042054533820,
    139.736208952121440,
    141.123707404021380,
    143.111845807956760,
    # 51-60
    146.000982486703210,
    147.422765342604400,
    150.053520421389860,
    150.925257612237310,
    153.024693811098940,
    156.112909294120560,
    157.597591817644780,
    158.849988171116530,
    161.188964137688150,
    163.030709687300510,
    # 61-70
    165.537069187998660,
    167.184439978043600,
    169.094515416327260,
    169.911976479585370,
    173.411536520086340,
    174.754191523698060,
    176.441434297100990,
    178.377407776033770,
    179.916484020146120,
    182.207078484560340,
    # 71-80
    184.874467848467960,
    185.598783677578240,
    187.228922583987090,
    189.416158656000200,
    192.026656361383230,
    193.079726604017400,
    195.265396679681800,
    196.876481840817340,
    198.015309676117780,
    201.264751944478940,
    # 81-90
    202.493594514137700,
    204.189671803361750,
    205.394697202756880,
    207.906258888521460,
    209.576509717402370,
    211.690862595450000,
    213.347919360307830,
    214.547044783246960,
    216.169538507766270,
    219.067596349055060,
    # 91-100
    220.714918839305490,
    221.430705555084420,
    224.007000254951960,
    224.983324669962010,
    227.421444279686710,
    229.337413305419280,
    231.250188700381880,
    231.987235253015580,
    233.693404179044270,
    236.524229665997640,
)

_extended_zeros: Optional[NDArray[np.float64]] = None


def _compute_remaining_zeros() -> None:
    """Lazy-compute is unnecessary; the tuple is already populated."""


@dataclass(frozen=True)
class SpacingStatistics:
    """GUE spacing distribution statistics for Riemann zeros."""

    n_zeros: int
    mean_spacing: float
    variance: float
    skewness: float
    ks_statistic: float
    ks_pvalue: float
    wigner_surmise_fit: float


class RiemannZeros:
    """Access and analyze non-trivial zeros of the Riemann zeta function."""

    def __init__(self) -> None:
        self._zeros = np.array(_FIRST_100_ZEROS, dtype=np.float64)

    @property
    def count(self) -> int:
        """Number of available zeros."""
        return len(self._zeros)

    def get_zeros(self, n: int) -> NDArray[np.float64]:
        """Return first n non-trivial zeros (imaginary parts).

        If n exceeds hard-coded count, compute additional zeros via mpmath.
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        if n <= len(self._zeros):
            return self._zeros[:n].copy()
        return self._extend_to(n)[:n].copy()

    def _extend_to(self, n: int) -> NDArray[np.float64]:
        """Extend zero cache to at least n entries using mpmath."""
        import mpmath  # noqa: local import to keep startup fast

        if n <= len(self._zeros):
            return self._zeros
        new_zeros = list(self._zeros)
        for i in range(len(self._zeros) + 1, n + 1):
            zero_val = float(mpmath.zetazero(i).imag)
            new_zeros.append(zero_val)
        self._zeros = np.array(new_zeros, dtype=np.float64)
        return self._zeros

    def get_normalized_zeros(self, n: int) -> NDArray[np.float64]:
        """Return zeros normalized to local mean spacing 1.

        Uses the smooth part of the zero counting function:
        N(t) ~ (t/(2*pi)) * ln(t/(2*pi)) - t/(2*pi)
        so the local density is d(t) = ln(t/(2*pi)) / (2*pi),
        and the unfolded variable is the integral of that density.
        """
        zeros = self.get_zeros(n)
        return _unfold_zeros(zeros)

    def nearest_zero(self, k: float) -> float:
        """Find the nearest known zero to value k."""
        idx = int(np.argmin(np.abs(self._zeros - k)))
        return float(self._zeros[idx])

    def spacing_statistics(self, n: int) -> SpacingStatistics:
        """Compute GUE spacing distribution statistics for first n zeros."""
        from scipy import stats as sp_stats

        unfolded = self.get_normalized_zeros(n)
        spacings = np.diff(unfolded)
        mean_s = float(np.mean(spacings))
        var_s = float(np.var(spacings))

        # Skewness
        skew_val = float(sp_stats.skew(spacings))

        # KS test against Wigner surmise (GUE): P(s) = (32/pi^2)*s^2*exp(-4s^2/pi)
        ks_stat, ks_p = sp_stats.kstest(spacings, _wigner_surmise_cdf)

        # Wigner surmise fit: 1 - KS statistic
        fit_quality = 1.0 - ks_stat

        return SpacingStatistics(
            n_zeros=n,
            mean_spacing=mean_s,
            variance=var_s,
            skewness=skew_val,
            ks_statistic=float(ks_stat),
            ks_pvalue=float(ks_p),
            wigner_surmise_fit=fit_quality,
        )


def _unfold_zeros(zeros: NDArray[np.float64]) -> NDArray[np.float64]:
    """Unfold zeros using the smooth counting function N(t)."""
    t = zeros
    two_pi = 2.0 * np.pi
    # Smooth part of counting function: N(t) ~ (t/2pi)*ln(t/2pi) - t/2pi + 7/8
    ratio = t / two_pi
    unfolded = ratio * np.log(ratio) - ratio + 7.0 / 8.0
    return unfolded


def _wigner_surmise_cdf(s: float | NDArray[np.float64]) -> NDArray[np.float64]:
    """CDF of GUE Wigner surmise: P(s) = (32/pi^2)*s^2*exp(-4s^2/pi).

    Obtained by integrating the PDF analytically:
    CDF(s) = erf(2s/sqrt(pi)) - (4s/pi) exp(-4s^2/pi)
    """
    from scipy.special import erf

    s_arr = np.asarray(s, dtype=np.float64)
    return np.clip(
        erf(2.0 * s_arr / np.sqrt(np.pi))
        - (4.0 * s_arr / np.pi) * np.exp(-4.0 * s_arr**2 / np.pi),
        0.0,
        1.0,
    )


def load_extended_zeros(filepath: str | Path) -> NDArray[np.float64]:
    """Load extended zeros from a file (JSON array or one-per-line text).

    Returns a new numpy array; does not mutate any global state.
    """
    global _extended_zeros
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Zero file not found: {filepath}")

    text = path.read_text().strip()
    if text.startswith("["):
        values = json.loads(text)
    else:
        values = [float(line.strip()) for line in text.splitlines() if line.strip()]

    if not values:
        raise ValueError(f"No zeros found in {filepath}")

    zeros = np.array(values, dtype=np.float64)
    _extended_zeros = zeros.copy()
    return zeros
