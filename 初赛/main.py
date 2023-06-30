import os
import time
import numpy as np
import scipy.sparse as sp
import scipy.optimize as so
import scipy.sparse.linalg as splg


def read_data(path):
    enter_people = []
    exit_people = []
    plant_path_num = []  # 存储星球卡口的前缀和
    plant_path_num.append(0)
    last_plant = -1
    with open(path + "/people_on_plant.txt", 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            plant_string, people_string = line.strip("\n").split(":")
            cur_enter_people, cur_exit_people = map(int, people_string.split(","))
            cur_plant, cur_path = map(int, plant_string.split("_"))

            cur_plant -= 1  # 从0开始
            if cur_plant != last_plant:
                plant_path_num.append(plant_path_num[-1])
                last_plant = cur_plant
            plant_path_num[-1] += 1
            enter_people.append(cur_enter_people)
            exit_people.append(cur_exit_people)

    result_max = []
    cur_row = 0
    row = []
    col = []
    value = []
    with open(path + "/path.txt", 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            _, port_string = line.strip("\n").split(":")
            port = port_string.split(",")
            min_people = float("inf")
            for i in range(len(port) // 2):
                cur_plant, cur_enter_col = map(int, port[2 * i].split("_"))
                cur_exit_col = int(port[2 * i + 1].split("_")[1])

                cur_plant -= 1
                enter_col = plant_path_num[cur_plant] + cur_enter_col - 1
                exit_col = plant_path_num[cur_plant] + plant_path_num[-1] + cur_exit_col - 1
                col.append(enter_col)
                row.append(cur_row)
                value.append(1)
                col.append(exit_col)
                row.append(cur_row)
                value.append(1)
                min_people = min(min_people, enter_people[enter_col], exit_people[exit_col - plant_path_num[-1]])
            cur_row += 1
            result_max.append(min_people)

    return row, col, value, enter_people + exit_people, cur_row, result_max


def write_data(path, result, result_max):
    with open(path + "/result.txt", 'w', encoding='utf-8') as fw:
        for i in range(len(result)):
            if result[i] < 1:
                result[i] = 1
            elif round(result[i]) > result_max[i]:
                result[i] = result_max[i]
            fw.write("path_" + str(i + 1) + ":" + str(round(result[i])) + "\n")


def write_data2(path, result):
    with open(path + "/result.txt", 'w', encoding='utf-8') as fw:
        for i in range(len(result)):
            fw.write("path_" + str(i + 1) + ":" + str(round(result[i] / 2)) + "\n")


def func(A, x, b):
    return 0.5 * np.linalg.norm(A.dot(x) - b) ** 2


def calc_grad(A, x, b, non_binding=None):
    Ax = A.dot(x) - b
    grad = A.T.dot(Ax)
    if non_binding is not None:
        grad[non_binding] = 0.0
    return grad


files_path = "../data/"
start_time = time.time()

for file in os.listdir(files_path):

    if time.time() - start_time > 59:
        break

    row, col, value, Y, path_num, result_max = read_data(files_path + file)

    if time.time() - start_time > 55:
        write_data2(files_path + file, result_max)
        continue

    Y = np.array(Y)
    # 构造稀疏矩阵
    csr = sp.csr_matrix((value, (col, row)), shape=(len(Y), path_num))
    # lsqr求解
    result = splg.lsqr(csr, Y, atol=1e-10, btol=1e-10)[0]
    result_max = np.array(result_max)
    # if np.min(result) < 1 or np.any(result > result_max+1):
    if np.min(result) < 1:
        # l-bfgs-b求解
        f_wrap = lambda x: func(csr, x, Y)
        jac_wrap = lambda x: calc_grad(csr, x, Y)
        out = so.minimize(f_wrap, np.ones(csr.shape[1]), jac=jac_wrap, tol=1e-8, method='L-BFGS-B',
                          bounds=[(1, result_max[i]) for i in range(csr.shape[1])],
                          options={'maxiter': 20000})
        result = out.x
        # print(file)
        # print(out.nit)
        # print(func(csr, np.round(result), Y))

    write_data(files_path + file, result, result_max)

# print("time:", time.time() - start_time)

