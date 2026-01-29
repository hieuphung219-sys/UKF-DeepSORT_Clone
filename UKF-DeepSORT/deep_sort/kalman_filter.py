# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class UnscentedKalmanFilter:
    def __init__(self):
        ndim = 7
        self._ndim = ndim
        self._no_sigma_points = 2 * ndim + 1
        self._lamda = 3 - (ndim + 2)
        self._update_mat = np.eye(2, ndim)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 100
        self._std_weight_acceleration = 1. / 100
        self.height = 0
        self.sigma_points = np.zeros((self._no_sigma_points + 4, self._ndim + 2))

    def initiate(self, measurement):
        # 1. Khởi tạo Mean 7 phần tử [x, y, a, h, vx, vy, vh]
        mean_pos = measurement[:4] 
        mean_vel = np.zeros(3) 
        mean = np.r_[mean_pos, mean_vel] 

        # 2. Khởi tạo độ lệch chuẩn (std) cho 7 phần tử
        # Chúng ta cần định nghĩa std cho cả 7 thành phần để tạo ma trận 7x7
        std = [
            2 * self._std_weight_position * measurement[3], # x
            2 * self._std_weight_position * measurement[3], # y
            1e-2,                                           # a (tỷ lệ khung hình ít biến động)
            2 * self._std_weight_position * measurement[3], # h
            10 * self._std_weight_velocity * measurement[3], # vx
            10 * self._std_weight_velocity * measurement[3], # vy
            10 * self._std_weight_velocity * measurement[3]  # vh
        ]
        
        self.height = measurement[3]
        # 3. Tạo ma trận hiệp phương sai 7x7 từ bình phương độ lệch chuẩn
        covariance = np.diag(np.square(std))
        
        print(f"[KIỂM TRA CUỐI] Mean shape: {mean.shape}, Cov shape: {covariance.shape}")
        return mean, covariance

    def generate_sigma_point(self, mean, covariance):
        # Sửa ndim + 2 thành 9 (vì ndim=7, cộng thêm 2 chiều nhiễu gia tốc)
        sigma_points = np.zeros((self._no_sigma_points + 4, self._ndim + 2)) 
        sigma_points[0] = mean
        L = np.linalg.cholesky(covariance)
        for i in range(0, self._ndim + 2): # Sửa ở đây
            sigma_points[i + 1] = mean + np.sqrt(self._ndim + 2 + self._lamda) * L[i]
            sigma_points[i + 1 + self._ndim + 2] = mean - np.sqrt(self._ndim + 2 + self._lamda) * L[i]
        return sigma_points

    def augmentation(self, mean, covariance):
        mean_aug = np.zeros(9) # Sửa 7 -> 9
        mean_aug[:7] = mean    # Sửa 5 -> 7
        covariance_aug = np.zeros((9, 9)) # Sửa 7x7 -> 9x9
        covariance_aug[:7, :7] = covariance # Sửa 5x5 -> 7x7
        
        # Giữ nguyên phần tính toán std cho acceleration
        std = [
            self._std_weight_acceleration * self.height,
            1e-6
        ]
        covariance_aug[7:, 7:] = np.diag(np.square(std)) # Sửa index 5 -> 7
        return mean_aug, covariance_aug
    
    def predict(self, mean, covariance):
        mean, covariance = self.augmentation(mean, covariance)
        sigma_points = self.generate_sigma_point(mean, covariance)
        predicted_sigma_points = np.zeros((self._no_sigma_points + 4, self._ndim))
        for i in range(self._no_sigma_points + 4):
            x = sigma_points[i]
            # x[0]=x, x[1]=y, x[2]=a, x[3]=h, x[4]=vx, x[5]=vy, x[6]=vh, x[7]=acc_noise_1, x[8]=acc_noise_2
            dt = 1.0 # Giả định delta t = 1 frame
            x[0] += x[4] * dt + 0.5 * x[7] * dt**2 # x = x + vx*t + 0.5*acc*t^2
            x[1] += x[5] * dt + 0.5 * x[8] * dt**2 # y = y + vy*t + 0.5*acc*t^2
            x[3] += x[6] * dt                      # h = h + vh*t
            # x[2] (tỷ lệ aspect ratio) giữ nguyên
            predicted_sigma_points[i] = x[:7] # Chỉ lấy 7 phần tử trạng thái gốc

        weights = np.zeros(self._no_sigma_points + 4)
        weights[0] = self._lamda / (self._ndim + 2 + self._lamda)
        weights[1:] = 0.5 / (self._ndim + 2 + self._lamda)
        mean = np.dot(weights, predicted_sigma_points)

        weights = np.diag(weights)
        covariance = np.linalg.multi_dot(
            ((mean.T - predicted_sigma_points).T, weights, (mean.T - predicted_sigma_points)))
        return mean, covariance, predicted_sigma_points

    def project(self, mean, covariance, height, predicted_sigma_points):
        # Sửa [:2] thành [:4] để lấy đủ x, y, a, h
        projected_sigma_points = predicted_sigma_points[:, :4].copy() 

        # Cập nhật số chiều cho weights
        weights = np.zeros(self._no_sigma_points + 4)
        weights[0] = self._lamda / (self._ndim + 2 + self._lamda)
        weights[1:] = 0.5 / (self._ndim + 2 + self._lamda)
        projected_mean = np.dot(weights, projected_sigma_points)

        weights_mat = np.diag(weights)
        delta = projected_sigma_points - projected_mean
        projected_covariance = np.linalg.multi_dot((delta.T, weights_mat, delta))
        
        # Thêm nhiễu đo lường cho cả 4 thành phần x, y, a, h
        std = [
            self._std_weight_position * height,
            self._std_weight_position * height,
            1e-1, # nhiễu cho aspect ratio
            self._std_weight_position * height
        ]
        innovation_cov = np.diag(np.square(std))
        
        # Tính toán Correlation (hiệp phương sai chéo)
        delta_x = predicted_sigma_points - mean
        correlation = np.linalg.multi_dot((delta_x.T, weights_mat, delta))
        
        return projected_mean, projected_covariance + innovation_cov, correlation

    def update(self, mean, covariance, measurement, predicted_sigma_points):
        projected_mean, projected_covariance, correlation = self.project(mean, covariance,
                                                                         measurement[3], predicted_sigma_points)
        kalman_gain = np.dot(correlation, np.linalg.inv(projected_covariance))

        innovation = measurement - projected_mean # measurement bây giờ có 4 phần tử

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_covariance, kalman_gain.T))
            
        # Tạm thời tắt hoặc sửa phần if check cũ vì nó đang dùng index cứng [:2]
        self.height = measurement[3]
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, height, predicted_sigma_points,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        mean, covariance, _ = self.project(mean, covariance, height, predicted_sigma_points)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
# Dán thêm vào deep_sort/kalman_filter.py

class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.
    The 8-dimensional state space is (x, y, a, h, vx, vy, va, vh).
    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T)) + innovation_cov
        return mean, covariance

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric='mahalanobis'):
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'mahalanobis':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha