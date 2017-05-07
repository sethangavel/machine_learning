import xlrd
import math
import csv
import numpy as np
from numpy.linalg import inv
from numpy.linalg import det

INDEX_SHEET = 0
INDEX_COL_GENDER = 0
INDEX_COL_HEIGHT = 1
INDEX_COL_HANDSPAN = 2
workbook = xlrd.open_workbook(
    '/Users/manoj/my/course/ML/Assignments/Assignment_2_Data_and_Template.xlsx')
query = [[69, 17.5], [66, 22], [70, 21.5], [69, 23.5]]
worksheet = workbook.sheet_by_index(INDEX_SHEET)


def write_CSV(filePath, data):
    with open(filePath, 'w') as outcsv:
        writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        for row in data:
            writer.writerow(row)
    outcsv.close()


def get_float_value(val):
    return float("{0:.4f}".format(val))


def get_all_gender_height_list():
    gender_list = [col_val for col_val in
                   worksheet.col_values(INDEX_COL_GENDER)]
    height_list = [col_val for col_val in
                   worksheet.col_values(INDEX_COL_HEIGHT)]
    handspan_list = [col_val for col_val in
                     worksheet.col_values(INDEX_COL_HANDSPAN)]
    return list(zip(gender_list, zip(height_list, handspan_list)))


def get_male_height_handspan_list(all_list):
    return [b for a, b in all_list if a == "Male"]


def get_female_height_handspan_list(all_list):
    return [b for a, b in all_list if a == "Female"]


def get_optimal_bin_count(min_samples):
    return get_optimal_bin_count_sturges(min_samples)


def get_optimal_bin_count_sturges(min_samples):
    return math.ceil(np.log2(min_samples) + 1)


def get_optimal_bin_count_rice(min_samples):
    return math.ceil(2 * get_cube_root(min_samples))


def get_cube_root(n):
    return n ** (1.0 / 3.0)


def get_bin(xi, xmin, xmax):
    return (round((bin_count - 1) *
                  ((xi - xmin) / (xmax - xmin))))


def get_histo_matrix_row_col(h, s):
    r = get_bin(h, min_height, max_height)
    c = get_bin(s, min_handspan, max_handspan)
    return [r, c]


def get_xi_for_row_col(row, col, row_width, col_width):
    # h = (row * (max_height - min_height) / (bin_count - 1)) + min_height
    # s = (col * (max_handspan - min_handspan) / (bin_count - 1)) + min_handspan
    h = min_height + (row * row_width) + (row_width / 2)
    s = min_handspan + (col * col_width) + (col_width / 2)
    return [h, s]


def get_histo_matrix(in_list):
    histo = np.zeros((bin_count, bin_count))
    for h, s in in_list:
        rc_tup = get_histo_matrix_row_col(h, s)
        histo[rc_tup[0]][rc_tup[1]] = histo[rc_tup[0]][rc_tup[1]] + 1
    return histo


def get_prob_by_histo(query_list):
    histo_prob = {}
    for h, s in query_list:
        rc_tup = get_histo_matrix_row_col(h, s)
        Hm_hs = male_hs_histo[rc_tup[0]][rc_tup[1]]
        Hf_hs = female_hs_histo[rc_tup[0]][rc_tup[1]]
        if Hm_hs + Hf_hs != 0:
            histo_prob[(h, s)] = Hf_hs / (Hm_hs + Hf_hs)
        else:
            histo_prob[(h, s)] = "NaN"
    return histo_prob


def get_mean_vector(vec):
    return np.mean(vec, axis=0, dtype=np.float64)


def get_covariance_matrix(vec):
    return np.cov(vec, rowvar=False, ddof=1)


def get_2d_pdf(xi, mu_x, cov_x):
    xi_mu = np.array([np.subtract(xi, mu_x)])
    cov_inverse = inv(cov_x)
    xi_mu_transpose = np.transpose(xi_mu)
    xi_scalar = (np.dot(xi_mu, cov_inverse).dot(xi_mu_transpose))
    part_2 = math.exp(-1 * xi_scalar / 2)
    cov_determinant = det(cov_x)
    part_1 = 1 / (2 * math.pi * math.pow(cov_determinant, 1 / 2))
    return part_1 * part_2


def get_post_probability_xi(xi, mu_f, cov_f, mu_m, cov_m):
    prior_prob_xi_given_f = get_2d_pdf(xi, mu_f, cov_f)
    prior_prob_xi_given_f_n = \
        len(female_height_handspan_list) * prior_prob_xi_given_f

    prior_prob_xi_given_m = get_2d_pdf(xi, mu_m, cov_m)
    prior_prob_xi_given_m_n = \
        len(male_height_handspan_list) * prior_prob_xi_given_m

    post_prob_f_given_xi = prior_prob_xi_given_f_n / (
        prior_prob_xi_given_f_n + prior_prob_xi_given_m_n)
    return post_prob_f_given_xi


def get_post_probability(vec, mu_f, cov_f, mu_m, cov_m):
    ret_prob = {}
    for xi in vec:
        ret_prob[tuple(xi)] = get_post_probability_xi(xi, mu_f, cov_f,
                                                      mu_m, cov_m)
    return ret_prob


def get_reconstructed_histo(mu_a, cov_a, mu_b, cov_b):
    histo = np.zeros((bin_count, bin_count))
    for r in range(0, bin_count):
        for c in range(0, bin_count):
            xi = get_xi_for_row_col(r, c)
            # print(xi)
            histo[r][c] = get_post_probability_xi(xi, mu_a, cov_a, mu_b, cov_b)
    return histo


def get_reconstructed_histo_pdf(mu_a, cov_a, total_samples, height_width,
                                span_width):
    print("BinWidth:", height_width, span_width)
    histo = np.zeros((bin_count, bin_count))
    for r in range(0, bin_count):
        for c in range(0, bin_count):
            xi = get_xi_for_row_col(r, c, height_width, span_width)
            # print(xi)
            histo[r][c] = total_samples * height_width * \
                                            span_width *  get_2d_pdf(xi, mu_a, cov_a)
    return histo


all_data = get_all_gender_height_list()

male_height_handspan_list = get_male_height_handspan_list(all_data)
male_height_list = [h for h, s in male_height_handspan_list]
male_handspan_list = [s for h, s in male_height_handspan_list]

female_height_handspan_list = get_female_height_handspan_list(all_data)
female_height_list = [h for h, s in female_height_handspan_list]
female_handspan_list = [s for h, s in female_height_handspan_list]

min_height = min(min(male_height_list), min(female_height_list))
max_height = max(max(male_height_list), max(female_height_list))
min_handspan = min(min(male_handspan_list), min(female_handspan_list))
max_handspan = max(max(male_handspan_list), max(female_handspan_list))

bin_count = get_optimal_bin_count(min(len(male_height_list),
                                      len(female_height_list)))

print("\nSample Size:")
print("F: {}".format(len(female_height_list)))
print("M: {}".format(len(male_height_list)))
print("Min height {}, Max height {}".format(min_height, max_height))
print("Min hand {}, Max hand {}".format(min_handspan, max_handspan))
print("Optimal bin count: {}".format(bin_count))

print("\nFemale Histogram")
female_hs_histo = get_histo_matrix(female_height_handspan_list)
print(female_hs_histo)
print("\nMale Histogram")
male_hs_histo = get_histo_matrix(male_height_handspan_list)
print(male_hs_histo)

print("\nPosterior probability of being Female given features:")
print(get_prob_by_histo(query))

print("\nFemale Mean vector:")
Mu_F = get_mean_vector(female_height_handspan_list)
print(Mu_F)

print("\nMale Mean vector:")
Mu_M = get_mean_vector(male_height_handspan_list)
print(Mu_M)

print("\nFemale covariance matrix:")
Cov_F = get_covariance_matrix(female_height_handspan_list)
print(Cov_F)

print("\nMale covariance matrix:")
Cov_M = get_covariance_matrix(male_height_handspan_list)
print(Cov_M)

print("\nPosterior probability of being Female given features:")
print(get_post_probability(query, Mu_F, Cov_F, Mu_M, Cov_M))

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

print("\nReconstructed Female Histo from PDF factors:")
# print(get_reconstructed_histo(Mu_F, Cov_F, Mu_M, Cov_M))
reconstructed_histo_f = get_reconstructed_histo_pdf(Mu_F, Cov_F,
                                                    len(female_height_handspan_list),
                                                    ((max_height - min_height) / bin_count),
                                                    ((max_handspan - min_handspan) / bin_count))
np.savetxt('out.csv', reconstructed_histo_f, fmt='%.20f,')
# write_CSV("reconstructed_histo_female.csv", reconstructed_histo_f)
print(repr(reconstructed_histo_f))

print("\nReconstructed Male Histo from PDF factors:")
# print(get_reconstructed_histo(Mu_M, Cov_M, Mu_F, Cov_F))
print(repr(get_reconstructed_histo_pdf(Mu_M, Cov_M,
                                  len(male_height_handspan_list),
                                  ((max_height - min_height) / bin_count),
                                  ((max_handspan - min_handspan) / bin_count))))
