import xlrd
import math
import numpy as np

BIN_COUNT = 32
INDEX_SHEET = 0
INDEX_COL_FEET = 0
INDEX_COL_INCH = 1
INDEX_COL_GENDER = 2
workbook = xlrd.open_workbook('/Users/manoj/my/course/ML/Assignment_1_Data_and_Template.xlsx')
query_height = [55, 60, 65, 70, 75, 80]
worksheet = workbook.sheet_by_index(INDEX_SHEET)


def get_float_value(val):
    return float("{0:.4f}".format(val))


def get_all_gender_height_list():
    feet_list = [col_val for col_val in worksheet.col_values(INDEX_COL_FEET)]
    inch_list = [col_val for col_val in worksheet.col_values(INDEX_COL_INCH)]
    gender_list = [col_val for col_val in worksheet.col_values(INDEX_COL_GENDER)]
    height_list = [(feet * 12 + inch) for feet, inch in zip(feet_list, inch_list)]
    return list(zip(gender_list, height_list))


def get_male_height_list(all_list):
    return [b for a, b in all_list if a == "Male"]


def get_female_height_list(all_list):
    return [b for a, b in all_list if a == "Female"]


def get_heights_by_bin_list(in_list):
    bin_list = [0] * BIN_COUNT
    for height in in_list:
        bin_list[get_bin(height)] += 1
    return bin_list


def get_bin(height):
    return (round((BIN_COUNT - 1) *
                  ((height - min_height) / (max_height - min_height))))


def get_female_height_prob_using_histo(male_height_histo, female_height_histo,
                                       query_height_list):
    return_prob = dict()
    for idx in range(0, len(query_height_list)):
        key = query_height_list[idx]
        bin_idx = get_bin(key)
        if male_height_histo[bin_idx] + female_height_histo[bin_idx] != 0:
            return_prob[key] = \
                get_float_value(female_height_histo[bin_idx] /
                                (male_height_histo[bin_idx] +
                                 female_height_histo[bin_idx]))
        else:
            return_prob[key] = "NaN"
    return return_prob


def get_mean(height_list):
    return np.mean(height_list)


def get_std_deviation(height_list):
    return np.std(height_list)


def get_pdf(height, std_dev, mean):
    part1 = (1 / (math.sqrt(2 * math.pi) * std_dev))
    part2 = (1 / 2) * pow(((height - mean) / std_dev), 2)
    pdf = part1 * math.exp(-1 * part2)
    return pdf


def get_female_height_prob_using_pdf(female_count, female_std_dev, female_mean,
                                     male_count, male_std_dec, male_mean,
                                     q_height_list):
    height_prob_map = dict()
    for idx in range(0, len(q_height_list)):
        key = q_height_list[idx]
        male_prob = male_count * get_pdf(key, male_std_dec,
                                         male_mean)
        female_prob = female_count * get_pdf(key, female_std_dev, female_mean)
        if female_prob + male_prob != 0:
            height_prob_map[key] = get_float_value(female_prob /
                                                   (female_prob + male_prob))
        else:
            height_prob_map[key] = "NaN"
    return height_prob_map

print("All Data")
all_data = get_all_gender_height_list()
male_list = get_male_height_list(all_data)
female_list = get_female_height_list(all_data)
min_height = min(min(male_list), min(female_list))
max_height = max(max(male_list), max(female_list))

male_height_bins = get_heights_by_bin_list(male_list)
female_height_bins = get_heights_by_bin_list(female_list)
print("\nMean Heights:")
print(get_float_value(get_mean(female_list)))
print(get_float_value(get_mean(male_list)))

print("\nMean Std Deviation:")
print(get_float_value(get_std_deviation(female_list)))
print(get_float_value(get_std_deviation(male_list)))

print("\nHistogram Result:")
print(get_female_height_prob_using_histo(male_height_bins, female_height_bins,
                                         query_height))

print("\nSample Size:")
print(len(female_list))
print(len(male_list))

print("\nBayes Theorem:")
print(get_female_height_prob_using_pdf(len(female_list), get_std_deviation(
    female_list), get_mean(female_list), len(male_list), get_std_deviation(
    male_list), get_mean(male_list), query_height))


partial_data = all_data[1:51]
male_list = get_male_height_list(partial_data)
female_list = get_female_height_list(partial_data)
min_height = min(min(male_list), min(female_list))
max_height = max(max(male_list), max(female_list))

print("\n\n")
print("Partial Data")
male_height_bins = get_heights_by_bin_list(male_list)
female_height_bins = get_heights_by_bin_list(female_list)

print("\nMean Male Heights:")
print(get_float_value(get_mean(female_list)))
print(get_float_value(get_mean(male_list)))

print("\nMean Std Deviation:")
print(get_std_deviation(female_list))
print(get_std_deviation(male_list))

print("\nHistogram Result:")
print(get_female_height_prob_using_histo(male_height_bins, female_height_bins,
                                         query_height))

print("\nSample Size:")
print(len(female_list))
print(len(male_list))

print("\nBayes Theorem:")
print(get_female_height_prob_using_pdf(len(female_list), get_std_deviation(
    female_list), get_mean(female_list), len(male_list), get_std_deviation(
    male_list), get_mean(male_list), query_height))



