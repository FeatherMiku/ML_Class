import csv
import random
import matplotlib.pyplot as plt


# 读取数据文件
def load_housing_data(filename):
    X, y = [], []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                # 提取特征列：RM, LSTAT, PTRATIO
                X.append([float(row['RM']), float(row['LSTAT']), float(row['PTRATIO'])])
                # 提取目标值：MEDV
                y.append(float(row['MEDV']))
            except ValueError:
                # 跳过无法转换为float的行或记录错误
                continue  # 或者打印错误日志等
    return X, y


# 构造三次多项式特征
def construct_polynomial_features(X, degree=3):
    """将输入特征扩展为三次多项式特征"""
    poly_features = []
    for row in X:
        poly_row = []
        for val in row:
            for d in range(1, degree + 1):
                poly_row.append(val ** d)
        poly_features.append(poly_row)
    return poly_features


# 初始化参数
def initialize_parameters(num_features):
    theta = [random.random() for _ in range(num_features)]  # 随机初始化权重
    b = random.random()  # 初始化偏置
    return theta, b


# 预测函数
def predict(X, theta, b):
    return [sum(X[i][j] * theta[j] for j in range(len(theta))) + b for i in range(len(X))]


# 损失函数：均方误差
def compute_loss(y_true, y_pred):
    m = len(y_true)
    return sum((y_pred[i] - y_true[i]) ** 2 for i in range(m)) / (2 * m)


# 计算梯度
def compute_gradients(X, y_true, y_pred, theta, b):
    m = len(y_true)
    d_theta = [0] * len(theta)
    d_b = 0

    # 计算每个参数的梯度
    for i in range(m):
        error = y_pred[i] - y_true[i]
        d_b += error
        for j in range(len(theta)):
            d_theta[j] += error * X[i][j]

    # 归一化梯度
    d_theta = [d / m for d in d_theta]
    d_b /= m

    return d_theta, d_b


# 梯度下降优化器
def gradient_descent(X, y, theta, b, learning_rate, num_iterations, tolerance=1e-6):
    prev_loss = float('inf')
    for i in range(num_iterations):
        # 预测
        y_pred = predict(X, theta, b)

        # 计算损失
        loss = compute_loss(y, y_pred)

        # 检查是否收敛
        if abs(loss - prev_loss) < tolerance:
            print(f"Converged at iteration {i} with loss: {loss}")
            break
        prev_loss = loss

        # 计算梯度
        d_theta, d_b = compute_gradients(X, y, y_pred, theta, b)

        # 更新参数
        theta = [theta[j] - learning_rate * d_theta[j] for j in range(len(theta))]
        b = b - learning_rate * d_b

        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {loss}")

    return theta, b


# 数据归一化
def normalize(X):
    num_features = len(X[0])
    min_vals = [min(column) for column in zip(*X)]
    max_vals = [max(column) for column in zip(*X)]

    X_normalized = [[(x - min_vals[i]) / (max_vals[i] - min_vals[i]) for x in column] for i, column in enumerate(zip(*X))]
    # 转置矩阵以恢复原始的行列排列
    X_normalized = [list(row) for row in zip(*X_normalized)]
    return X_normalized


# 读取和预处理数据
X, y = load_housing_data('housing.csv')
X = normalize(X)  # 归一化输入数据
X_poly = construct_polynomial_features(X, degree=3)  # 三次多项式扩展

# 初始化参数
theta, b = initialize_parameters(len(X_poly[0]))

# 设置梯度下降超参数
learning_rate = 0.01
num_iterations = 1000

# 通过梯度下降训练模型
theta, b = gradient_descent(X_poly, y, theta, b, learning_rate, num_iterations)

# 用训练好的模型进行预测
y_pred = predict(X_poly, theta, b)

# 计算最终损失
final_loss = compute_loss(y, y_pred)
print(f"Final Loss: {final_loss}")


def plot_predictions(y_true, y_pred):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 6))
    indices = list(range(len(y_true)))
    plt.plot(indices, y_true, label='实际价格', color='blue', marker='o')
    plt.plot(indices, y_pred, label='预测价格', color='red', linestyle='--', marker='x')
    plt.xlabel('样本索引')
    plt.ylabel('价格')
    plt.title('实际房价与预测房价的比较')
    plt.legend()
    plt.grid(True)
    plt.show()


plot_predictions(y, y_pred)