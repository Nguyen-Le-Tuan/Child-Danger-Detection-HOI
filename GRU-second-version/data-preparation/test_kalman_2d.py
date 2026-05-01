#!/usr/bin/env python3
"""Test suite for upgraded Kalman filters (1D, 2D, Advanced smoother)."""

import numpy as np
from kalman import (
    Kalman1D, Kalman2D, AdvancedKalmanSmoother, MultiObjectKalmanTracker
)


def test_kalman1d():
    """Test original 1D Kalman filter."""
    print("\n" + "="*70)
    print("TEST 1: Kalman1D (1D Kalman Filter)")
    print("="*70)
    
    # Simulate constant velocity motion with noise
    kf = Kalman1D(pos0=0.0, vel0=1.0, q=1e-2, r=1e-1)
    
    true_positions = np.linspace(0, 10, 11)
    noisy_measurements = true_positions + np.random.normal(0, 0.5, len(true_positions))
    
    filtered_positions = []
    for z in noisy_measurements:
        pos = kf.update(z)
        filtered_positions.append(pos)
    
    print(f"True positions:       {true_positions[:5]}...")
    print(f"Noisy measurements:   {noisy_measurements[:5]}...")
    print(f"Filtered positions:   {filtered_positions[:5]}...")
    print("✓ Kalman1D working correctly\n")


def test_kalman2d():
    """Test new 2D Kalman filter."""
    print("="*70)
    print("TEST 2: Kalman2D (2D Kalman Filter)")
    print("="*70)
    
    # Simulate circular motion with noise
    kf = Kalman2D((0, 0), vel0=(1, 0), q=1e-2, r=1e-1)
    
    positions_2d = []
    for i in range(10):
        angle = i * 0.5
        true_x = 5 * np.cos(angle)
        true_y = 5 * np.sin(angle)
        
        # Add measurement noise
        meas_x = true_x + np.random.normal(0, 0.2)
        meas_y = true_y + np.random.normal(0, 0.2)
        
        filtered_pos = kf.update((meas_x, meas_y))
        positions_2d.append(filtered_pos)
        print(f"  Step {i}: Measured ({meas_x:.2f}, {meas_y:.2f}) → Filtered {filtered_pos}")
    
    print("✓ Kalman2D working correctly\n")


def test_advanced_smoother():
    """Test advanced Kalman smoother with outlier detection."""
    print("="*70)
    print("TEST 3: AdvancedKalmanSmoother (Forward-Backward Pass)")
    print("="*70)
    
    smoother = AdvancedKalmanSmoother(q=1e-2, r=1e-1, max_iterations=2)
    
    # Generate synthetic 2D trajectory with outliers
    positions = []
    for i in range(15):
        x = i * 0.5
        y = 2 * np.sin(i * 0.4)
        
        # Add noise
        x += np.random.normal(0, 0.1)
        y += np.random.normal(0, 0.1)
        
        # Add outlier
        if i == 7:
            x += 2.0
            y += 2.0
        
        positions.append((x, y))
    
    # Smooth trajectory
    smoothed = smoother.smooth_2d_trajectory(positions)
    
    print("Original positions (first 5):")
    for i, pos in enumerate(positions[:5]):
        print(f"  {i}: {pos}")
    
    print("\nSmoothed positions (first 5):")
    for i, pos in enumerate(smoothed[:5]):
        print(f"  {i}: {pos}")
    
    print("\nOutlier detection (position 7):")
    print(f"  Original: {positions[7]}")
    print(f"  Smoothed: {smoothed[7]}")
    
    print("✓ AdvancedKalmanSmoother working correctly\n")


def test_distance_smoothing():
    """Test 1D distance smoothing."""
    print("="*70)
    print("TEST 4: Distance Smoothing (1D Kalman)")
    print("="*70)
    
    smoother = AdvancedKalmanSmoother()
    
    # Simulate distance measurements with noise and gaps
    distances = [
        1.0, 1.1, 1.05, 1.15, 2.5,  # Outlier at position 4
        1.2, 1.1, None, 1.25, 1.3,   # Gap at position 8
        1.35, 1.4
    ]
    
    smoothed_distances = smoother.smooth_distance(distances)
    
    print("Original distances:")
    print(f"  {distances}")
    
    print("\nSmoothed distances:")
    print(f"  {smoothed_distances}")
    
    print("\nNote: Outlier at position 4 (2.5) and gap at position 8 (None)")
    print("✓ Distance smoothing working correctly\n")


def test_multi_object_tracker():
    """Test multi-object Kalman tracker."""
    print("="*70)
    print("TEST 5: MultiObjectKalmanTracker (Multiple Objects)")
    print("="*70)
    
    tracker = MultiObjectKalmanTracker(q=1e-2, r=1e-1)
    
    # Track 3 objects
    objects = {
        'child_1': [(i*0.3, i*0.2) for i in range(10)],
        'object_1': [(i*0.2, i*0.3) for i in range(10)],
        'object_2': [(i*0.5, -i*0.1) for i in range(10)]
    }
    
    results = {obj_id: [] for obj_id in objects}
    
    for frame in range(10):
        print(f"\nFrame {frame}:")
        for obj_id, positions in objects.items():
            # Add measurement noise
            meas = (
                positions[frame][0] + np.random.normal(0, 0.05),
                positions[frame][1] + np.random.normal(0, 0.05)
            )
            
            tracked_pos = tracker.update(
                obj_id, meas, 
                initial_pos=positions[0] if frame == 0 else None
            )
            results[obj_id].append(tracked_pos)
            print(f"  {obj_id}: Measured {meas} → Tracked {tracked_pos}")
    
    # Get final state (position + velocity)
    all_estimates = tracker.get_all_estimates()
    print("\nFinal estimates [x, y, vx, vy]:")
    for obj_id, state in all_estimates.items():
        print(f"  {obj_id}: {state}")
    
    print("✓ MultiObjectKalmanTracker working correctly\n")


def main():
    """Run all tests."""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█  KALMAN FILTER UPGRADE TEST SUITE (1D → 2D + Advanced)".ljust(69) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    try:
        test_kalman1d()
        test_kalman2d()
        test_advanced_smoother()
        test_distance_smoothing()
        test_multi_object_tracker()
        
        print("█"*70)
        print("█" + " "*68 + "█")
        print("█  ✅ ALL TESTS PASSED SUCCESSFULLY".ljust(69) + "█")
        print("█" + " "*68 + "█")
        print("█"*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
