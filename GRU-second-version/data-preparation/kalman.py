import numpy as np
from collections import deque


class Kalman1D:
    """1D Kalman filter for smooth trajectory interpolation.
    
    Uses constant-velocity model with process noise (q) and measurement noise (r).
    Suitable for filling missing bbox coordinates.
    """
    
    def __init__(self, pos0, vel0=0.0, q=1e-2, r=1e-1, dt=1.0):
        """Initialize 1D Kalman filter.
        
        Args:
            pos0: initial position
            vel0: initial velocity (default 0)
            q: process noise covariance (larger = more dynamic motion)
            r: measurement noise covariance (larger = less trust measurements)
            dt: time step between frames (default 1)
        """
        self.x = np.array([float(pos0), float(vel0)], dtype=float)
        self.P = np.eye(2)
        self.F = np.array([[1, dt], [0, 1]], dtype=float)  # State transition
        self.H = np.array([[1, 0]], dtype=float)  # Measurement matrix
        self.Q = np.array([
            [q*dt**3/3, q*dt**2/2],
            [q*dt**2/2, q*dt]
        ], dtype=float)  # Process noise
        self.R = np.array([[r]], dtype=float)  # Measurement noise

    def predict(self):
        """Predict next state without measurement."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0])

    def update(self, z):
        """Update state with measurement.
        
        Args:
            z: measurement value
        
        Returns:
            float: updated position estimate
        """
        if z is None or np.isnan(z):
            return self.predict()
        
        z = float(z)
        y = z - (self.H @ self.x)[0]  # Innovation
        S = (self.H @ self.P @ self.H.T + self.R)[0, 0]  # Innovation covariance
        K = (self.P @ self.H.T) / S  # Kalman gain
        
        self.x = self.x + K.flatten() * y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        
        return float(self.x[0])


class Kalman2D:
    """2D Kalman filter for smooth 2D trajectory interpolation.
    
    Uses constant-velocity model for both x and y coordinates.
    State: [x, y, vx, vy] (position and velocity in 2D).
    """
    
    def __init__(self, pos0, vel0=(0.0, 0.0), q=1e-2, r=1e-1, dt=1.0):
        """Initialize 2D Kalman filter.
        
        Args:
            pos0: initial position (x, y) tuple or array
            vel0: initial velocity (vx, vy) tuple or array (default (0, 0))
            q: process noise covariance
            r: measurement noise covariance
            dt: time step between frames (default 1)
        """
        pos0 = np.array(pos0, dtype=float).flatten()
        vel0 = np.array(vel0, dtype=float).flatten()
        
        # State: [x, y, vx, vy]
        self.x = np.array([pos0[0], pos0[1], vel0[0], vel0[1]], dtype=float)
        self.P = np.eye(4)
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)
        
        # Measurement matrix (measure only position)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)
        
        # Process noise covariance
        self.Q = np.array([
            [q*dt**3/3, 0, q*dt**2/2, 0],
            [0, q*dt**3/3, 0, q*dt**2/2],
            [q*dt**2/2, 0, q*dt, 0],
            [0, q*dt**2/2, 0, q*dt]
        ], dtype=float)
        
        # Measurement noise covariance
        self.R = np.array([
            [r, 0],
            [0, r]
        ], dtype=float)

    def predict(self):
        """Predict next state without measurement.
        
        Returns:
            tuple: predicted (x, y) position
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return (float(self.x[0]), float(self.x[1]))

    def update(self, z):
        """Update state with 2D measurement.
        
        Args:
            z: measurement (x, y) tuple or array, or None for missing data
        
        Returns:
            tuple: updated (x, y) position estimate
        """
        if z is None:
            return self.predict()
        
        z = np.array(z, dtype=float).flatten()
        
        # Check for NaN values
        if np.any(np.isnan(z)):
            return self.predict()
        
        # Innovation (measurement residual)
        z_pred = self.H @ self.x
        y = z - z_pred
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        return (float(self.x[0]), float(self.x[1]))

    def get_state(self):
        """Get current state [x, y, vx, vy]."""
        return self.x.copy()

    def get_velocity(self):
        """Get current velocity (vx, vy)."""
        return (float(self.x[2]), float(self.x[3]))


class AdvancedKalmanSmoother:
    """Advanced Kalman smoother combining forward-backward pass and outlier detection.
    
    Implements:
    1. Forward Kalman filter pass
    2. Backward Rauch-Tsiokdis smoother pass
    3. Adaptive outlier detection using Mahalanobis distance
    4. Multi-pass smoothing for convergence
    """
    
    def __init__(self, q=1e-2, r=1e-1, dt=1.0, max_iterations=2):
        """Initialize advanced smoother.
        
        Args:
            q: process noise covariance
            r: measurement noise covariance
            dt: time step between frames
            max_iterations: number of smoothing iterations (2-3 recommended)
        """
        self.q = q
        self.r = r
        self.dt = dt
        self.max_iterations = max_iterations

    def smooth_2d_trajectory(self, positions, velocities=None):
        """Smooth 2D trajectory using forward-backward pass.
        
        Args:
            positions: list of (x, y) positions (may contain None for missing)
            velocities: optional list of initial velocity estimates
        
        Returns:
            list: smoothed (x, y) positions
        """
        n = len(positions)
        if n == 0:
            return []
        
        # Find valid measurements
        valid_idx = [i for i, pos in enumerate(positions) if pos is not None]
        if not valid_idx:
            return [None] * n
        
        # Initialize filtered states
        filtered_states = [None] * n
        filtered_covs = [None] * n
        
        # Forward pass with outlier detection
        initial_vel = velocities[valid_idx[0]] if velocities else (0.0, 0.0)
        kalman = Kalman2D(
            positions[valid_idx[0]],
            vel0=initial_vel,
            q=self.q,
            r=self.r,
            dt=self.dt
        )
        
        filtered_states[valid_idx[0]] = kalman.get_state().copy()
        filtered_covs[valid_idx[0]] = kalman.P.copy()
        
        for i in range(1, n):
            # Predict or update
            if i in valid_idx:
                # Detect outliers using Mahalanobis distance
                predicted = kalman.predict()
                
                # Calculate Mahalanobis distance
                innovation = np.array(positions[i]) - np.array(predicted)
                pred_cov = kalman.P[:2, :2]
                
                try:
                    mahal_dist = np.sqrt(
                        innovation @ np.linalg.inv(pred_cov) @ innovation.T
                    )
                    
                    # Outlier threshold (3-sigma)
                    if mahal_dist > 3.0:
                        # Treat as outlier, only predict
                        pos = kalman.predict()
                    else:
                        pos = kalman.update(positions[i])
                except:
                    # Fallback if matrix inversion fails
                    pos = kalman.update(positions[i])
            else:
                pos = kalman.predict()
            
            filtered_states[i] = kalman.get_state().copy()
            filtered_covs[i] = kalman.P.copy()
        
        # Backward Rauch-Tsitsiklis smoother pass
        smoothed_states = self._backward_smooth(filtered_states, filtered_covs)
        
        # Extract positions
        smoothed_pos = [
            (float(state[0]), float(state[1])) if state is not None else None
            for state in smoothed_states
        ]
        
        return smoothed_pos

    def _backward_smooth(self, filtered_states, filtered_covs):
        """Backward Rauch-Tsitsiklis smoothing pass.
        
        Args:
            filtered_states: forward pass states
            filtered_covs: forward pass covariances
        
        Returns:
            list: smoothed states
        """
        n = len(filtered_states)
        smoothed_states = [state.copy() if state is not None else None 
                          for state in filtered_states]
        
        # Start from end and work backward
        for i in range(n - 2, -1, -1):
            if filtered_states[i] is None or filtered_states[i+1] is None:
                continue
            
            # Transition matrix
            F = np.array([
                [1, 0, self.dt, 0],
                [0, 1, 0, self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=float)
            
            # Predicted covariance at i+1
            P_pred = F @ filtered_covs[i] @ F.T
            
            try:
                # Smoother gain
                C = filtered_covs[i] @ F.T @ np.linalg.inv(P_pred)
                
                # Smoothed state
                smoothed_states[i] = (
                    filtered_states[i] + 
                    C @ (smoothed_states[i+1] - F @ filtered_states[i])
                )
            except:
                pass  # Keep filtered state if inversion fails
        
        return smoothed_states

    def smooth_distance(self, distances):
        """Smooth distance values using 1D Kalman filter.
        
        Args:
            distances: list of distance values (may contain None)
        
        Returns:
            list: smoothed distance values
        """
        valid_dist = [d for d in distances if d is not None]
        if not valid_dist:
            return distances
        
        # Initialize with first valid measurement
        first_valid_idx = distances.index(next(d for d in distances if d is not None))
        kalman = Kalman1D(distances[first_valid_idx], q=self.q, r=self.r, dt=self.dt)
        
        smoothed = [None] * len(distances)
        smoothed[first_valid_idx] = distances[first_valid_idx]
        
        for i in range(first_valid_idx + 1, len(distances)):
            if distances[i] is not None:
                smoothed[i] = kalman.update(distances[i])
            else:
                smoothed[i] = kalman.predict()
        
        return smoothed


class MultiObjectKalmanTracker:
    """Tracks multiple objects using 2D Kalman filter with data association.
    
    Manages separate Kalman filters for each object/child pair.
    """
    
    def __init__(self, q=1e-2, r=1e-1, dt=1.0):
        """Initialize multi-object tracker.
        
        Args:
            q: process noise covariance
            r: measurement noise covariance
            dt: time step between frames
        """
        self.q = q
        self.r = r
        self.dt = dt
        self.filters = {}  # Dictionary: object_id -> Kalman2D

    def get_or_create_filter(self, obj_id, initial_pos, initial_vel=(0.0, 0.0)):
        """Get existing filter or create new one.
        
        Args:
            obj_id: unique object identifier
            initial_pos: (x, y) initial position
            initial_vel: (vx, vy) initial velocity estimate
        
        Returns:
            Kalman2D: filter for this object
        """
        if obj_id not in self.filters:
            self.filters[obj_id] = Kalman2D(
                initial_pos,
                vel0=initial_vel,
                q=self.q,
                r=self.r,
                dt=self.dt
            )
        return self.filters[obj_id]

    def update(self, obj_id, measurement, initial_pos=None, initial_vel=(0.0, 0.0)):
        """Update tracker with measurement.
        
        Args:
            obj_id: unique object identifier
            measurement: (x, y) position or None for missing data
            initial_pos: position if creating new track
            initial_vel: initial velocity estimate
        
        Returns:
            tuple: updated (x, y) position estimate
        """
        if obj_id not in self.filters and initial_pos is not None:
            self.get_or_create_filter(obj_id, initial_pos, initial_vel)
        elif obj_id not in self.filters:
            return None
        
        kalman = self.filters[obj_id]
        if measurement is None:
            return kalman.predict()
        else:
            return kalman.update(measurement)

    def get_all_estimates(self):
        """Get current estimates for all tracked objects.
        
        Returns:
            dict: object_id -> (x, y, vx, vy) state
        """
        return {
            obj_id: kalman.get_state()
            for obj_id, kalman in self.filters.items()
        }