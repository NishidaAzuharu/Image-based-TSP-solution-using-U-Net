import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os

 
def gray_show_route(X_list, Y_list, C_route):
    img_size = (256, 256) 

    img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255 

    X_pixels = (np.array(X_list) * (img_size[0] - 1)).astype(int)
    Y_pixels = (np.array(Y_list) * (img_size[1] - 1)).astype(int)

    Y_pixels = img_size[1] - 1 - Y_pixels

    line_width = 3

    for i in range(len(C_route) - 1):
        pt1 = (X_pixels[C_route[i]], Y_pixels[C_route[i]])
        pt2 = (X_pixels[C_route[i + 1]], Y_pixels[C_route[i + 1]])
        cv2.line(img, pt1, pt2, (0, 0, 0), line_width)

    pt1 = (X_pixels[C_route[-1]], Y_pixels[C_route[-1]])
    pt2 = (X_pixels[C_route[0]], Y_pixels[C_route[0]])
    cv2.line(img, pt1, pt2, (0, 0, 0), line_width)

    vertices_R = 2
    for x, y in zip(X_pixels, Y_pixels):
        cv2.circle(img, (x, y), vertices_R, (0, 0, 0), -1)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



    return gray_img

def gray_show_vertices(X_list, Y_list):
    
    img_size = (256, 256)

    img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255 

    X_pixels = (np.array(X_list) * (img_size[0] - 1)).astype(int)
    Y_pixels = (np.array(Y_list) * (img_size[1] - 1)).astype(int)

    Y_pixels = img_size[1] - 1 - Y_pixels

    vertices_R = 2 
    for x, y in zip(X_pixels, Y_pixels):
        cv2.circle(img, (x, y), vertices_R, (0, 0, 0), -1) 

    vertices_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    return vertices_img

def calc_node_cords(X_list, Y_list):
    li = []
    li.append(X_list)
    li.append(Y_list)
    return np.array(li).T

def calc_adj_matrix(route):
    n = len(route)
    adj_matrix = np.zeros((n, n), dtype=int)
    for i in range(n-1):
        idx_1 = route[i]
        idx_2 = route[i+1]

        adj_matrix[idx_1, idx_2] = 1
        adj_matrix[idx_2, idx_1] = 1
    adj_matrix[idx_2, route[0]] = 1
    adj_matrix[route[0], idx_2] = 1
    return adj_matrix

def str2list_floats(input_string):
    if type(input_string) == list:
        return input_string
    else:
        matches = re.findall(r"[-e0-9.]+", input_string)  # 正規表現を使用して数字、"-", "e", "."を抽出
        return [int(match) for match in matches] 

def get_args():
    parser = argparse.ArgumentParser(description="for preprocess settings")


    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="input file path"
    )

    parser.add_argument(
        "--num_data",
        type=int,
        default=87000,
        help="Number of data to be used"
    )

    args = parser.parse_args()
    return args

def main(args):
    path = args.input_file
    num_data = args.num_data

    lines = []
    with open(path, "r") as  fp:
        for _ in range(num_data):
            line = fp.readline()
            if not line:
                break
            lines.append(line)


    in_data = []
    out_data = []
    num_node_list = []
    for line in tqdm(lines, total=num_data):
        line = line.strip().split(" ")
        num_node = int(line[1])

        X_list = []
        Y_list = []

        for j in range(0, 2*num_node, 2):
            X_list.append(float(line[2+j]))
            Y_list.append(float(line[2+j+1]))
        route=list(map(int, line[-num_node:]))

        in_data.append(gray_show_vertices(X_list, Y_list))
        out_data.append(gray_show_route(X_list, Y_list, route))
        num_node_list.append(num_node)

    
    new_df = pd.DataFrame(columns=["input_img", "output_img", "num_node"])
    in_series_data = pd.Series(in_data)
    out_series_data = pd.Series(out_data)
    num_node_seies = pd.Series(num_node_list)
    new_df["input_img"] = in_series_data
    new_df["output_img"] = out_series_data
    new_df["num_node"] = num_node_seies

    train_df, test_df = train_test_split(new_df, test_size=0.2, random_state=42)

    
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)


    print("save train...")
    train_df.to_pickle("resources/train_data.pkl")
    print("save test...")
    test_df.to_pickle("resources/test_data.pkl")
    print("Done")

if __name__ == "__main__":
    args = get_args()
    main(args)


