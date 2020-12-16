from skimage import color
import sys
import glob
from pathlib import Path
import numpy as np
from PIL import Image
import itertools
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import csv


LAMBDA = 0.5
T = 0.38

def load_img(img_path):
	out_np = np.asarray(Image.open(img_path))
	if(out_np.ndim==2):
		out_np = np.tile(out_np[:,:,None],3)
	return out_np


def resize_img(img, HW=(256,256), resample=3):
	return np.asarray(Image.fromarray(img).resize((HW[1],HW[0]), resample=resample))


def estimate_color(z):
	""" z is a vector of size (B, Q, H, W) """

	color_dict = {}
	with open('color.csv', 'r') as infile:
		for line in infile:
			split_line = line.rstrip('\n').split(',')
			index, row, col = split_line
			color_dict[index] = (row, col)


	B, Q, H, W = z.shape
	exp_log_arr = np.exp(np.log(z) / T)
	sum_arr = np.sum(exp_log_arr, axis = 1)
	sum_arr = np.expand_dims(sum_arr, axis=1)
	sum_arr = np.repeat(sum_arr, Q, axis=1)
	ft_arr = exp_log_arr / sum_arr

	ft_arr.reshape(B, H, W, Q)

	res_arr = np.zeros((B, H, W, 2))

	for b in range(B):
		for h in range(H):
			for w in range(W):
				weighted_x = 0
				weighted_y = 0
				for q in range(Q):
					row, col = color_dict[q]
					prob = ft_arr[b][h][w][q]
					x = row * 10 + 5
					y = col * 10 + 5
					weighted_x += x * prob
					weighted_y += y * prob

				res_arr[b][h][w] = [weighted_x, weighted_y]

	return res_arr




def main():
	img_pathes = glob.glob("D:\\Project_Assets\\VOCdevkit\\VOC2012\\JPEGImages\\*")

	# ab_ranges_list = np.array(np.meshgrid([i for i in range(-11, 11)], [i for i in range(-11, 11)])).T.reshape(484, 2)
	# ab_ranges = [tuple(pair) for pair in ab_ranges_list]
	# global_dict = {key: 0 for key in ab_ranges}

	bin_distri = np.zeros((22, 22))

	Q_count = 0
	for index, img_path in enumerate(img_pathes):
		if True:
			img_rs = resize_img(load_img(img_path))
			img_lab_rs = color.rgb2lab(img_rs)

			img_ab_rs = img_lab_rs[:,:,1:]
			img_ab_rs_d = img_ab_rs // 10

			for i in range(img_ab_rs_d.shape[0]):
				for j in range(img_ab_rs_d.shape[1]):
					ab_tuple = tuple(img_ab_rs_d[i][j])
					if bin_distri[int(ab_tuple[0] + 11)][int(ab_tuple[1] + 11)] == 0:
						Q_count += 1
					bin_distri[int(ab_tuple[0] + 11)][int(ab_tuple[1] + 11)] += 1
			print("#Image", index, "Currently Observed Color Channel:", Q_count)
		else:
			break

	#print(bin_distri)
	mask = bin_distri == 0
	bin_distri /= np.sum(bin_distri)
	plt.subplot(141)
	plt.imshow(bin_distri, cmap='hot')

	gaussian_smoothed_distri = gaussian_filter(bin_distri, sigma=0.5)
	#print(gaussian_smoothed_distri)
	plt.subplot(142)
	plt.imshow(gaussian_smoothed_distri, cmap='hot')

	copy_gaussian_distri = np.copy(gaussian_smoothed_distri)
	copy_gaussian_distri[mask] = float('inf')

	mixed_distri = 1 / ((1 - LAMBDA) * copy_gaussian_distri + LAMBDA / Q_count)
	#print(mixed_distri)
	plt.subplot(143)
	plt.imshow(mixed_distri, cmap='hot')

	normalized_factor = np.sum(gaussian_smoothed_distri * mixed_distri)
	#print(normalized_factor)
	weight_distri = mixed_distri / normalized_factor
	print(weight_distri)
	Q = np.count_nonzero(weight_distri > 0)
	print(Q)
	print(np.sum(gaussian_smoothed_distri * weight_distri))
	plt.subplot(144)
	plt.imshow(weight_distri, cmap='hot')

	plt.show()

	index = 0
	for i in range(weight_distri.shape[0]):
		for j in range(weight_distri.shape[1]):
			if weight_distri[i][j] > 0:
				with open('color.csv', mode='w', newline='', encoding='utf-8-sig') as current_file:
					data_writer = csv.writer(current_file)
					data_writer.writerow(
                            [index, i, j, weight_distri[i][j]]
                        )
				index += 1





if __name__ == "__main__":
	main()
