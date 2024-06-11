import pandas as pd
import numpy as np
import random
import math

# Sampling function
def Partition(number_of_sample,limit_array):
    coefficient_lower = np.zeros((number_of_sample, 2))
    coefficient_upper = np.zeros((number_of_sample, 2))
    for i in range(number_of_sample):
        coefficient_lower[i, 0] = 1 - i / number_of_sample
        coefficient_lower[i, 1] = i / number_of_sample
    for i in range(number_of_sample):
        coefficient_upper[i, 0] = 1 - (i + 1) / number_of_sample
        coefficient_upper[i, 1] = (i + 1) / number_of_sample
    partition_lower = coefficient_lower @ limit_array.T
    partition_upper = coefficient_upper @ limit_array.T
    partition_range = np.dstack((partition_lower.T, partition_upper.T))
    return partition_range

def Representative(partition_range):
    number_of_value = partition_range.shape[0]
    numbers_of_row = partition_range.shape[1]
    coefficient_random = np.zeros((number_of_value, numbers_of_row, 2))
    representative_random = np.zeros((numbers_of_row, number_of_value))
    for m in range(number_of_value):
        for i in range(numbers_of_row):
            y = random.random()
            coefficient_random[m, i, 0] = 1 - y
            coefficient_random[m, i, 1] = y
    temp_arr = partition_range * coefficient_random
    for j in range(number_of_value):
        temp_random = temp_arr[j, :, 0] + temp_arr[j, :, 1]
        representative_random[:, j] = temp_random
    return representative_random

def Rearrange(arr_random):
    for i in range(arr_random.shape[1]):
        np.random.shuffle(arr_random[:, i])
    return arr_random

def ParameterArray(limitArray, sampleNumber):
    arr = Partition(sampleNumber, limitArray)
    parametersMatrix = Rearrange(Representative(arr))
    return parametersMatrix

def process_benchmark_functions(benchmark_functions):
    for name, props in benchmark_functions.items():
        dim = props["dim"]
        arr_limit = props["limit"]
        calc = props["calc"]

        for k in range(1, 11):
            size = k * (dim + 1) * (dim + 2)
            arr = ParameterArray(arr_limit, size)
            response = calc(arr)
            output_name = f"{name}_{k}"
            Output(arr, response, output_name)


# Output function
def Output(X,y,name):
    filename = name + ".csv"
    sample = np.c_[X,y]
    sample_data = pd.DataFrame(sample)
    sample_data.to_csv(filename)


def calc_BF4(arr):
    a = [[3.0, 10.0, 30.0], [0.1, 10.0, 35.0], [3.0, 10.0, 30.0], [0.1, 10.0, 35.0]]
    c = [1.0, 1.2, 3.0, 3.2]
    p = [[0.3689, 0.1170, 0.2673], [0.4699, 0.4387, 0.7470], [0.1091, 0.8732, 0.5547], [0.2815, 0.5743, 0.8828]]
    response = []
    for x in arr:
        y = 0
        for m in range(4):
            temp1 = 0
            for n in range(3):
                temp1 = temp1 - a[m][n] * (x[n] - p[m][n]) ** 2
            y = y - c[m] * math.exp(temp1)
        response.append(y)
    return response

def calc_BF5(arr):
    a = [[10.0, 3.0, 17.0, 3.5, 1.7, 8.0], [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
         [3.0, 3.5, 1.7, 10.0, 17.0, 8.0], [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]]
    c = [1.0, 1.2, 3.0, 3.2]
    p = [[0.1312, 0.1696, 0.5529, 0.0124, 0.8283, 0.5886], [0.2329, 0.4135, 0.4135, 0.3736, 0.1004, 0.9991],
         [0.2348, 0.1415, 0.1451, 0.2883, 0.3047, 0.6650], [0.4047, 0.8828, 0.8828, 0.5741, 0.1091, 0.0382]]
    response = []
    for x in arr:
        y = 0
        for m in range(4):
            temp1 = 0
            for n in range(6):
                temp1 = temp1 - a[m][n] * (x[n] - p[m][n]) ** 2
            y = y - c[m] * math.exp(temp1)
        response.append(y)
    return response




# Benchmark function for building knowledge graph
benchmark_functions_KG = {
    "BF1": {
        "dim": 2,
        "limit": np.array([[-5, 10], [0, 15]]).T,
        "calc": lambda arr: [
            (x2 - 5.1*(x1**2)/(4*math.pi*math.pi) + 5*x1/math.pi - 6)**2 + 10*(1-1/(8*math.pi))*math.cos(x1) + 10
            for x1, x2 in arr
        ]
    },
    "BF2": {
        "dim": 2,
        "limit": np.array([[-3, 3], [-2, 2]]).T,
        "calc": lambda arr: [
            (4 - 2.1 * (x1 ** 2) + (x1 ** 4) / 3) ** 2 + x1 * x2 + (-4 + 4 * (x2 ** 2)) * (x2 ** 2)
            for x1, x2 in arr
        ]
    },
    "BF3": {
        "dim": 2,
        "limit": np.array([[-2, 2], [-2, 2]]).T,
        "calc": lambda arr: [
            (1 + ((x1 + x2 +1)**2)*(19 - 4*x1 +3*x1*x1 - 14*x2 + 6*x1*x2 + 3*x2*x2)) *
            (30 + (2*x1 - 3*x2)**2) * (18 - 32*x1 + 12*x1*x1 + 48*x2 - 36*x1*x2 + 27*x2*x2)
            for x1, x2 in arr
        ]
    },
    "BF4": {
        "dim": 3,
        "limit": np.array([[-1, -1, -1], [1, 1, 1]]).T,
        "calc": calc_BF4
    },
    "BF5": {
        "dim": 6,
        "limit": np.array([[-1, -1, -1, -1, -1, -1], [1, 1, 1, 1, 1, 1]]).T,
        "calc": calc_BF5
    },
    "BF6": {
        "dim": 3,
        "limit": np.array([[-3, -3, -3], [3, 3, 3]]).T,
        "calc": lambda arr: [
            (x1-1)**2 + (x1-x2)**2 + (x2-x3)**2
            for x1, x2, x3 in arr
        ]
    },
    "BF7": {
        "dim": 4,
        "limit": np.array([[0, 0, 0, 0], [15, 15, 15, 15]]).T,
        "calc": lambda arr: [
            x1**2 + x2**2 + x1*x2 - 14*x1 - 16*x2 + (x3-10)**2 + 4*(x4-5)**2
            for x1, x2, x3, x4 in arr
        ]
    },
    "BF8": {
        "dim": 5,
        "limit": np.array([[2.1 for _ in range(5)], [9.9 for _ in range(5)]]).T,
        "calc": lambda arr: [
            sum((np.log(x[j] - 2)**2 + np.log(10 - x[j])**2) for j in range(5)) -
            np.prod([x[j]**2 for j in range(5)])
            for x in arr
        ]
    },
    "BF9": {
        "dim": 10,
        "limit": np.array([[2.1 for _ in range(10)], [9.9 for _ in range(10)]]).T,
        "calc": lambda arr: [
            sum((np.log(x[j] - 2)**2 + np.log(10 - x[j])**2) for j in range(10)) -
            np.prod([x[j]**2 for j in range(10)])
            for x in arr
        ]
    },
    "BF10": {
        "dim": 10,
        "limit": np.array([[0 for _ in range(10)], [15 for _ in range(10)]]).T,
        "calc": lambda arr: [
            sum(x[j]**2 + (x[j] - x[j-1])**2 - 14*x[j-1] - 16*x[j] for j in range(1, 10)) +
            (x[0]-10)**2 + 4*(x[1]-5)**2 + (x[2]-3)**2 + 2*(x[3]-1)**2 + 5*x[4]**2 + 7*(x[5]-11)**2 +
            2*(x[6]-10)**2 + (x[7]-7)**2 + 45
            for x in arr
        ]
    },
    "BF11": {
        "dim": 3,
        "limit": np.array([[-32.768 for _ in range(3)], [32.768 for _ in range(3)]]).T,
        "calc": lambda arr: [
            -20 * math.exp(-0.20 * math.sqrt(sum(x[j]**2 for j in range(3))/3))
            for x in arr
        ]
    },
    "BF12": {
        "dim": 6,
        "limit": np.array([[-32.768 for _ in range(6)], [32.768 for _ in range(6)]]).T,
        "calc": lambda arr: [
            -20 * math.exp(-0.20 * math.sqrt(sum(x[j]**2 for j in range(6))/6))
            for x in arr
        ]
    },
    "BF13": {
        "dim": 9,
        "limit": np.array([[-32.768 for _ in range(9)], [32.768 for _ in range(9)]]).T,
        "calc": lambda arr: [
            -20 * math.exp(-0.20 * math.sqrt(sum(x[j]**2 for j in range(9))/9))
            for x in arr
        ]
    },
    "BF14": {
        "dim": 2,
        "limit": np.array([[-5.12 for _ in range(2)], [5.12 for _ in range(2)]]).T,
        "calc": lambda arr: [
            - (1 + math.cos(12*math.sqrt(x1**2 + x2**2))) / (2 + 0.50*(x1**2)*(x2**2))
            for x1, x2 in arr
        ]
    },
    "BF15": {
        "dim": 2,
        "limit": np.array([[-512 for _ in range(2)], [512 for _ in range(2)]]).T,
        "calc": lambda arr: [
            - (x2+47)*math.sin(math.sqrt(abs(x2+ (x1/2) + 47))) - x1*math.sin(math.sqrt(abs(x1 - (x2 +47))))
            for x1, x2 in arr
        ]
    },
    "BF16": {
        "dim": 2,
        "limit": np.array([[-5.12 for _ in range(2)], [5.12 for _ in range(2)]]).T,
        "calc": lambda arr: [
            sum(i*math.cos((i + 1)*x1 + 1) for i in range(1, 6)) *
            sum(i*math.cos((i + 1) * x2 + i) for i in range(1, 6))
            for x1, x2 in arr
        ]
    },
    "BF17": {
        "dim": 4,
        "limit": np.array([[-600 for _ in range(4)], [600 for _ in range(4)]]).T,
        "calc": lambda arr: [
            1 + sum(x[j]**2/4000 for j in range(4)) - np.prod([math.cos(x[j]/math.sqrt(j+1)) for j in range(4)])
            for x in arr
        ]
    },
    "BF18": {
        "dim": 7,
        "limit": np.array([[-600 for _ in range(7)], [600 for _ in range(7)]]).T,
        "calc": lambda arr: [
            1 + sum(x[j]**2/4000 for j in range(7)) - np.prod([math.cos(x[j]/math.sqrt(j+1)) for j in range(7)])
            for x in arr
        ]
    },
    "BF19": {
        "dim": 10,
        "limit": np.array([[-600 for _ in range(10)], [600 for _ in range(10)]]).T,
        "calc": lambda arr: [
            1 + sum(x[j]**2/4000 for j in range(10)) - np.prod([math.cos(x[j]/math.sqrt(j+1)) for j in range(10)])
            for x in arr
        ]
    },
    "BF20": {
        "dim": 2,
        "limit": np.array([[-10 for _ in range(2)], [10 for _ in range(2)]]).T,
        "calc": lambda arr: [
            - 0.0001*(abs(math.sin(x1)*math.sin(x2)*math.exp(abs(100 - math.sqrt(x1**2 + x2**2)/math.pi))) + 1)**0.10
            for x1, x2 in arr
        ]
    },
    "BF21": {
        "dim": 2,
        "limit": np.array([[-10 for _ in range(2)], [10 for _ in range(2)]]).T,
        "calc": lambda arr: [
            (math.sin(3*math.pi*x1))**2 + ((x1-1)**2)*(1+ (math.sin(3*math.pi*x2))**2) + ((x2-1)**2)*(1+ (math.sin(2*math.pi*x2))**2)
            for x1, x2 in arr
        ]
    },
    "BF22": {
        "dim": 4,
        "limit": np.array([[-5.12 for _ in range(4)], [5.12 for _ in range(4)]]).T,
        "calc": lambda arr: [
            10*4 + sum(x[j]**2 - 10*math.cos(2*math.pi*x[j]) for j in range(4))
            for x in arr
        ]
    },
    "BF23": {
        "dim": 7,
        "limit": np.array([[-5.12 for _ in range(7)], [5.12 for _ in range(7)]]).T,
        "calc": lambda arr: [
            10*7 + sum(x[j]**2 - 10*math.cos(2*math.pi*x[j]) for j in range(7))
            for x in arr
        ]
    },
    "BF24": {
        "dim": 10,
        "limit": np.array([[-5.12 for _ in range(10)], [5.12 for _ in range(10)]]).T,
        "calc": lambda arr: [
            10*10 + sum(x[j]**2 - 10*math.cos(2*math.pi*x[j]) for j in range(10))
            for x in arr
        ]
    },
    "BF25": {
        "dim": 2,
        "limit": np.array([[-100 for _ in range(2)], [100 for _ in range(2)]]).T,
        "calc": lambda arr: [
            0.50 + (math.sin(x1**2-x2**2)**2 - 0.50)/(1+0.001*(x1**2+x2**2))**2
            for x1, x2 in arr
        ]
    },
    "BF26": {
        "dim": 2,
        "limit": np.array([[-100 for _ in range(2)], [100 for _ in range(2)]]).T,
        "calc": lambda arr: [
            0.50 + (math.cos(math.sin(abs(x1**2-x2**2))) - 0.50)/(1+0.001*(x1**2+x2**2))**2
            for x1, x2 in arr
        ]
    },
    "BF27": {
        "dim": 4,
        "limit": np.array([[-500 for _ in range(4)], [500 for _ in range(4)]]).T,
        "calc": lambda arr: [
            418.98*4 + sum(x[j]*math.sin(math.sqrt(abs(x[j]))) for j in range(4))
            for x in arr
        ]
    },
    "BF28": {
        "dim": 7,
        "limit": np.array([[-500 for _ in range(7)], [500 for _ in range(7)]]).T,
        "calc": lambda arr: [
            418.98*7 + sum(x[j]*math.sin(math.sqrt(abs(x[j]))) for j in range(7))
            for x in arr
        ]
    },
    "BF29": {
        "dim": 10,
        "limit": np.array([[-500 for _ in range(10)], [500 for _ in range(10)]]).T,
        "calc": lambda arr: [
            418.98*10 + sum(x[j]*math.sin(math.sqrt(abs(x[j]))) for j in range(10))
            for x in arr
        ]
    },
    "BF30": {
        "dim": 2,
        "limit": np.array([[-10 for _ in range(2)], [10 for _ in range(2)]]).T,
        "calc": lambda arr: [
            -abs(math.sin(x1)*math.cos(x2)*math.exp(11-math.sqrt(x1**2+x2**2)/math.pi))
            for x1, x2 in arr
        ]
    },
    "BF31": {
        "dim": 2,
        "limit": np.array([[-100 for _ in range(2)], [100 for _ in range(2)]]).T,
        "calc": lambda arr: [
            x1**2 + 2*x2**2 - 0.30*math.cos(3*math.pi*x1) - 0.40*math.cos(4*math.pi*x2) - 0.70
            for x1, x2 in arr
        ]
    },
    "BF32": {
        "dim": 2,
        "limit": np.array([[-100 for _ in range(2)], [100 for _ in range(2)]]).T,
        "calc": lambda arr: [
            x1**2 + 2*x2**2 - 0.30*math.cos(3*math.pi*x1)*math.cos(4*math.pi*x2)+0.30
            for x1, x2 in arr
        ]
    },
    "BF33": {
        "dim": 2,
        "limit": np.array([[-100 for _ in range(2)], [100 for _ in range(2)]]).T,
        "calc": lambda arr: [
            x1**2 + 2*x2**2 - 0.30*math.cos(3*math.pi*x1+4*math.pi*x2)+0.30
            for x1, x2 in arr
        ]
    },
    "BF34": {
        "dim": 2,
        "limit": np.array([[-2 for _ in range(2)], [2 for _ in range(2)]]).T,
        "calc": lambda arr: [
            sum(((j+10)*(x[j-1]**i - 1/j**i))**2 for i in range(1,3) for j in range(1,3))
            for x in arr
        ]
    },
    "BF35": {
        "dim": 5,
        "limit": np.array([[-65.54 for _ in range(5)], [65.54 for _ in range(5)]]).T,
        "calc": lambda arr: [
            sum(sum(x[l]**2 for l in range(j+1)) for j in range(5))
            for x in arr
        ]
    },
    "BF36": {
        "dim": 8,
        "limit": np.array([[-65.54 for _ in range(8)], [65.54 for _ in range(8)]]).T,
        "calc": lambda arr: [
            sum(sum(x[l]**2 for l in range(j+1)) for j in range(8))
            for x in arr
        ]
    },
    "BF37": {
        "dim": 5,
        "limit": np.array([[-5.12 for _ in range(5)], [5.12 for _ in range(5)]]).T,
        "calc": lambda arr: [
            sum(x[j]**2 for j in range(5))
            for x in arr
        ]
    },
    "BF38": {
        "dim": 8,
        "limit": np.array([[-5.12 for _ in range(8)], [5.12 for _ in range(8)]]).T,
        "calc": lambda arr: [
            sum(x[j]**2 for j in range(8))
            for x in arr
        ]
    },
    "BF39": {
        "dim": 2,
        "limit": np.array([[-10 for _ in range(2)], [10 for _ in range(2)]]).T,
        "calc": lambda arr: [
            (x1+2*x2-7)**2 + (2*x1+x2-5)**2
            for x1, x2 in arr
        ]
    },
    "BF40": {
        "dim": 2,
        "limit": np.array([[-10 for _ in range(2)], [10 for _ in range(2)]]).T,
        "calc": lambda arr: [
            0.26*(x1**2+x2**2) - 0.48*x1*x2
            for x1, x2 in arr
        ]
    },
    "BF41": {
        "dim": 2,
        "limit": np.array([[-1.5, -3], [4, 4]]).T,
        "calc": lambda arr: [
            math.sin(x1+x2) + (x1-x2)**2 - 1.5*x1 + 2.5*x2 + 1
            for x1, x2 in arr
        ]
    },
    "BF42": {
        "dim": 2,
        "limit": np.array([[-5 for _ in range(2)], [5 for _ in range(2)]]).T,
        "calc": lambda arr: [
            2*x1**2 - 1.05*x1**4 + x1**6/6 + x1*x2 + x2**2
            for x1, x2 in arr
        ]
    },
    "BF43": {
        "dim": 2,
        "limit": np.array([[-3, -2], [3, 2]]).T,
        "calc": lambda arr: [
            (4-2.1*x1**2+x1**4/3)*x1**2 + x1*x2 + (-4+4*x2**2)*x2**2
            for x1, x2 in arr
        ]
    },
    "BF44": {
        "dim": 2,
        "limit": np.array([[0 for _ in range(2)], [math.pi for _ in range(2)]]).T,
        "calc": lambda arr: [
            sum(-math.sin(x[j])*(math.sin((j+1)*x[j]**2/math.pi))**20 for j in range(2))
            for x in arr
        ]
    },
    "BF45": {
        "dim": 5,
        "limit": np.array([[0 for _ in range(5)], [math.pi for _ in range(5)]]).T,
        "calc": lambda arr: [
            sum(-math.sin(x[j])*(math.sin((j+1)*x[j]**2/math.pi))**20 for j in range(5))
            for x in arr
        ]
    },
    "BF46": {
        "dim": 8,
        "limit": np.array([[0 for _ in range(8)], [math.pi for _ in range(8)]]).T,
        "calc": lambda arr: [
            sum(-math.sin(x[j])*(math.sin((j+1)*x[j]**2/math.pi))**20 for j in range(8))
            for x in arr
        ]
    }
}


# Benchmark function for verification
benchmark_functions_verification = {
    "Perm": {
        "dim": 10,
        "limit": np.array([[-1, -2, -3, -4, -5, -6, -7, -8, -9, -10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).T,
        "calc": lambda arr: [
            sum(sum(((((j+1)**(l+1) + 0.5)**2)*((arr[i][j]/(j+1))**(l+1) - 1))**2 for j in range(l+1)) for l in range(arr.shape[1]))
            for i in range(arr.shape[0])
        ]
    },
    "Dixon and Price": {
        "dim": 5,
        "limit": np.array([[-10 for _ in range(5)], [10 for _ in range(5)]]).T,
        "calc": lambda arr: [
            (arr[i][0]-1)**2 + sum((j+1)*(arr[i][j]**2 - arr[i][j-1])**2 for j in range(1, arr.shape[1]))
            for i in range(arr.shape[0])
        ]
    },
    "Beale": {
        "dim": 2,
        "limit": np.array([[-4.5 for _ in range(2)], [4.5 for _ in range(2)]]).T,
        "calc": lambda arr: [
            (1.5-x1+x1*x2)**2 + (2.25-x1+x1*x2**2)**2 + (2.625-x1+x1*x2**3)**2
            for x1, x2 in arr
        ]
    }
}


process_benchmark_functions(benchmark_functions_KG)
process_benchmark_functions(benchmark_functions_verification)

