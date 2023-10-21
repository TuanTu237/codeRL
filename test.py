import numpy as np
import matplotlib.pyplot as plt

# Số lượng bandit (k)
k = 10

# Xác suất chọn một bandit ngẫu nhiên
epsilon = 0.1

# Giá trị thực sự của các bandit (không biết trong thực tế)
true_bandit_values = np.random.normal(0, 1, k)

# Khởi tạo Q(a) và N(a)
Q = np.zeros(k)
N = np.zeros(k)

# Lưu trữ phần thưởng theo thời gian để vẽ biểu đồ
reward_history = []

# Số lượt chơi/iterations
num_rounds = 1000

for t in range(1, num_rounds + 1):
    if np.random.rand() < epsilon:
        # Khám phá: Chọn một bandit ngẫu nhiên với xác suất epsilon
        A = np.random.choice(k)
    else:
        # Khai thác: Chọn bandit có giá trị ước tính cao nhất (phá vỡ sự cân bằng bằng cách chọn ngẫu nhiên nếu có nhiều bandit có giá trị bằng nhau)
        max_Q = np.max(Q)
        best_bandits = np.where(Q == max_Q)[0]
        A = np.random.choice(best_bandits)

    # Mô phỏng phần thưởng cho bandit đã chọn (giá trị thực sự không biết trong thực tế)
    reward = np.random.normal(true_bandit_values[A], 1)

    # Cập nhật Q(a) và N(a)
    N[A] += 1
    Q[A] = Q[A] + (reward - Q[A]) / N[A]

    # Theo dõi phần thưởng cho mỗi lượt chơi
    reward_history.append(reward)

# Tính tỷ lệ thắng trung bình tích luỹ
cumulative_average_reward = np.cumsum(reward_history) / np.arange(1, num_rounds + 1)

# Vẽ biểu đồ tỷ lệ thắng trung bình tích luỹ
plt.plot(range(1, num_rounds + 1), cumulative_average_reward)
plt.xlabel("Số lượt chơi")
plt.ylabel("Tỷ lệ thắng trung bình tích luỹ")
plt.title("Giải thuật Bandit Đơn giản")
plt.show()