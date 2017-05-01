import xlrd
import math
import numpy as np

INDEX_SHEET = 0
INDEX_COL_GENDER = 0
INDEX_COL_HEIGHT = 1
INDEX_COL_HANDSPAN = 2
workbook = xlrd.open_workbook('/Users/manoj/my/course/ML/Assignment_2_Data_and_Template.xlsx')
query_height = []
worksheet = workbook.sheet_by_index(INDEX_SHEET)


def get_float_value(val):
    return float("{0:.4f}".format(val))


def get_all_gender_height_list():
    gender_list = [col_val for col_val in worksheet.col_values(INDEX_COL_GENDER)]
    height_list = [col_val for col_val in worksheet.col_values(INDEX_COL_HEIGHT)]
    handspan_list = [col_val for col_val in worksheet.col_values(INDEX_COL_HANDSPAN)]
    return list(zip(gender_list, zip(height_list, handspan_list)))


def get_male_height_handspan_list(all_list):
    return [b for a, b in all_list if a == "Male"]


def get_female_height_handspan_list(all_list):
    return [b for a, b in all_list if a == "Female"]


def get_optimal_bin_count(min_samples):
    return math.ceil(np.log2(min_samples) + 1)


def get_bin(xi, xmin, xmax):
    return (round((bin_count - 1) *
                  ((xi - xmin) / (xmax - xmin))))


def get_histo_matrix_row_col(h, s):
    r = get_bin(h, min_height, max_height)
    c = get_bin(s, min_handspan, max_handspan)
    return [r, c]


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
        histo_prob[(h, s)] = female_hs_histo[rc_tup[0]][rc_tup[1]] / (
            male_hs_histo[rc_tup[0]][rc_tup[1]] +
            female_hs_histo[rc_tup[0]][rc_tup[1]])
    return histo_prob

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


query = [[69, 17.5], [66, 22], [70, 21.5], [69, 23.5]]
print(get_prob_by_histo(query))

