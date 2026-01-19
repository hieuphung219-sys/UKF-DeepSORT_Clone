# vim: expandtab:ts=4:sw=4
class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.
    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = 1
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age
        # --- BỔ SUNG KHỞI TẠO BIẾN CHO UKF ---
        self.predicted_sigma_points = None

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.
        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.
        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        """
        # --- SỬA LỖI TẠI DÒNG NÀY: HỨNG 3 GIÁ TRỊ ---
        self.mean, self.covariance, self.predicted_sigma_points = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.
        """
        # Cập nhật dùng thêm predicted_sigma_points
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah(), self.predicted_sigma_points)
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == 1 and self.hits >= self._n_init:
            self.state = 2

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == 1:
            self.state = 3
        elif self.time_since_update > self._max_age:
            self.state = 3

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == 1

    def is_confirmed(self):
        """Returns True if this track is confirmed.
        """
        return self.state == 2

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted.
        """
        return self.state == 3