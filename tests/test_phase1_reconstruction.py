import numpy as np
import pytest

def calculate_planarity_rmse(points: np.ndarray) -> float:
    """
    Phase 1 Gate Metric: Calculate the Root Mean Square Error (RMSE) of points 
    to their best-fit plane using Singular Value Decomposition (SVD).
    """
    if len(points) < 3:
        return 0.0

    # Center the points
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    # Compute SVD
    _, _, vh = np.linalg.svd(centered_points)
    
    # The normal of the best-fit plane is the last row of V^T
    normal = vh[-1, :]

    # Calculate distances from points to the plane
    # Distance = (p - centroid) dot normal
    distances = np.dot(centered_points, normal)
    
    # Calculate RMSE in meters, return in centimeters
    rmse_meters = np.sqrt(np.mean(distances**2))
    return rmse_meters * 100.0

def test_planarity_gate_criteria_perfect_plane():
    """Phase 1: A perfect synthetic floor should have 0 cm error."""
    # Create a perfectly flat 10x10 grid on the Y=0 plane
    x, z = np.meshgrid(np.linspace(0, 10, 10), np.linspace(0, 10, 10))
    y = np.zeros_like(x)
    perfect_floor = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    
    rmse = calculate_planarity_rmse(perfect_floor)
    assert rmse < 1e-5, f"Perfect plane RMSE should be near 0, got {rmse}"

def test_planarity_gate_criteria_noisy_reconstruction():
    """Phase 1: A realistic noisy floor should pass the < 3.0 cm gate criteria."""
    rng = np.random.default_rng(42)
    x, z = np.meshgrid(np.linspace(0, 5, 20), np.linspace(0, 5, 20))
    
    # Simulate neural depth noise: +/- 2.5 cm uniform noise
    y_noise = rng.uniform(-0.025, 0.025, size=x.shape)
    
    noisy_floor = np.stack([x.flatten(), y_noise.flatten(), z.flatten()], axis=1)
    
    rmse = calculate_planarity_rmse(noisy_floor)
    
    # Gate criteria: RMSE must be under 3.0 cm
    assert rmse < 3.0, f"Planarity Gate Failed: RMSE {rmse:.2f} cm > 3.0 cm limit. Bilateral smoothing required."
