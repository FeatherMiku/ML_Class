import matplotlib.pyplot as plt
import math

x = [i * 2 * math.pi / 100 for i in range(100)]  # 生成0-2pi之间的100个点
y_actual = [math.sin(xi) for xi in x]  # 计算sin(x)的实际值


# 构造五次多项式特征矩阵
def construct_polynomial_features(x, degree=5):
    X = [[xi ** j for j in range(degree + 1)] for xi in x]
    return X


# 矩阵转置
def transpose(matrix):
    return [list(row) for row in zip(*matrix)]


# 矩阵乘法
def matmul(matrix_a, matrix_b):
    result = [[0 for _ in range(len(matrix_b[0]))] for _ in range(len(matrix_a))]
    for i in range(len(matrix_a)):
        for j in range(len(matrix_b[0])):
            result[i][j] = sum(matrix_a[i][k] * matrix_b[k][j] for k in range(len(matrix_a[0])))
    return result


# 高斯-约当消去法求逆矩阵
def gauss_jordan_inverse(A):
    n = len(A)
    # 创建增广矩阵[A | I]
    augmented_matrix = [A[i] + [float(i == j) for j in range(n)] for i in range(n)]

    # 高斯消元过程
    for i in range(n):
        # 选主元
        pivot_row = i
        for k in range(i + 1, n):
            if abs(augmented_matrix[k][i]) > abs(augmented_matrix[pivot_row][i]):
                pivot_row = k
        augmented_matrix[i], augmented_matrix[pivot_row] = augmented_matrix[pivot_row], augmented_matrix[i]

        pivot = augmented_matrix[i][i]
        if pivot == 0:
            raise ZeroDivisionError("矩阵不可逆，因为主元为零")

        # 归一化当前行
        for j in range(i, 2 * n):
            augmented_matrix[i][j] /= pivot

        # 使用当前行消去其他行
        for k in range(n):
            if k != i:
                factor = augmented_matrix[k][i]
                for j in range(i, 2 * n):
                    augmented_matrix[k][j] -= factor * augmented_matrix[i][j]

    # 提取单位矩阵部分作为逆矩阵
    inverse_matrix = [[row[j] for j in range(n, 2 * n)] for row in augmented_matrix]
    return inverse_matrix


# 正规方程((X^T X)θ=X^T y)
def linear_regression(X, y):
    X_transpose = transpose(X)  # X 的转置
    A = matmul(X_transpose, X)  # A = X^T * X
    b = matmul(X_transpose, [[yi] for yi in y])  # b = X^T * y, y 是列向量

    A_inv = gauss_jordan_inverse(A)

    theta = matmul(A_inv, b)
    return [t[0] for t in theta]


# 使用计算的参数进行预测
def predict(X, theta):
    return [sum(X[i][j] * theta[j] for j in range(len(theta))) for i in range(len(X))]


# 定义损失函数（均方误差）
def compute_loss(y_true, y_pred):
    return sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true))) / len(y_true)


# 构造多项式特征矩阵
X = construct_polynomial_features(x, degree=5)

# 计算线性回归参数
theta = linear_regression(X, y_actual)

# 用模型预测结果
y_pred = predict(X, theta)

# 计算损失
loss = compute_loss(y_actual, y_pred)

# 打印计算的参数和损失值
print("计算的参数 theta:", theta)
print("均方误差 (MSE):", loss)

# 绘制预测曲线与实际曲线的对比图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(x, y_actual, label='真实 sin(x)', color='blue')
plt.plot(x, y_pred, label='预测多项式', color='red', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Sin(x) vs 多项式回归')
plt.show()
