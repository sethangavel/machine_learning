import xlrd
import math

BIN_COUNT = 32
workbook = xlrd.open_workbook('/Users/manoj/my/course/ML/Assignment_1_Data_and_Template.xlsx')
query_height = [55, 60, 65, 70, 75, 80]
worksheet = workbook.sheet_by_index(0)


def get_all_list():
    feet_list = [col_val for col_val in worksheet.col_values(0)]
    inch_list = [col_val for col_val in worksheet.col_values(1)]
    gender_list = [col_val for col_val in worksheet.col_values(2)]
    height_list = [(feet * 12 + inch) for feet, inch in zip(feet_list, inch_list)]
    return list(zip(gender_list, height_list))


def get_male_list(all_list):
    return [b for a, b in all_list if a == "Male"]


def get_female_list(all_list):
    return [b for a, b in all_list if a == "Female"]


def compute_bin(in_list):
    return_list = [0] * BIN_COUNT
    for height in in_list:
        return_list[get_bin(height)] += 1
    return return_list


def get_bin(height):
    return (round((BIN_COUNT - 1) *
                  ((height - min_height) / (max_height - min_height))))


def get_female_height_probability(male_height_histo, female_height_histo,
                                  query_height_list):
    return_prob = [0] * len(query_height_list)
    for idx in range(0, len(query_height_list)):
        bin_idx = get_bin(query_height_list[idx])
        if male_height_histo[bin_idx] + female_height_histo[bin_idx] != 0:
            return_prob[idx] = \
                female_height_histo[bin_idx] / (male_height_histo[bin_idx] +
                                                female_height_histo[bin_idx])
    return return_prob


def get_mean(height_list):
    return sum(height_list) / len(height_list)


def get_std_deviation(height_list):
    mean = get_mean(height_list)
    sqr_dist = 0
    for ht in height_list:
        sqr_dist += pow(abs(mean - ht), 2)
    variance = sqr_dist / len(height_list)
    std_dev = math.sqrt(variance)
    return std_dev


def get_pdf(height, std_dev, mean):
    part1 = (1 / (math.sqrt(2 * math.pi) * std_dev))
    part2 = (1 / 2) * pow(((height - mean) / std_dev), 2)
    pdf = part1 * math.exp(-1 * part2)
    return pdf


def get_female_height_prob_pdf(female_count, female_std_dev, female_mean,
                               male_count, male_std_dec, male_mean,
                               q_height_list):
    return_prob = [0] * len(q_height_list)
    for idx in range(0, len(q_height_list)):
        male_prob = male_count * get_pdf(q_height_list[idx], male_std_dec,
                                         male_mean)
        female_prob = female_count * get_pdf(q_height_list[idx], female_std_dev,
                                             female_mean)
        if female_prob + male_prob != 0:
            return_prob[idx] = female_prob / (female_prob + male_prob)
    return return_prob

print("All Data")
all_data = get_all_list()
male_list = get_male_list(all_data)
female_list = get_female_list(all_data)
min_height = min(min(male_list), min(female_list))
max_height = max(max(male_list), max(female_list))

male_height_bins = compute_bin(male_list)
female_height_bins = compute_bin(female_list)
# print(female_height_bins)
# print(male_height_bins)
print("\nMean Male Heights:")
print(get_mean(female_list))
print(get_mean(male_list))

print("\nMean Std Deviation:")
print(get_std_deviation(female_list))
print(get_std_deviation(male_list))

print("\nHistogram Result:")
print(get_female_height_probability(male_height_bins, female_height_bins,
                                    query_height))

print("\nSample Size:")
print(len(female_list))
print(len(male_list))

print("\nBaysian:")
print(get_female_height_prob_pdf(len(female_list), get_std_deviation(
    female_list), get_mean(female_list), len(male_list), get_std_deviation(
    male_list), get_mean(male_list), query_height))


partial_data = all_data[1:51]
male_list = get_male_list(partial_data)
female_list = get_female_list(partial_data)
min_height = min(min(male_list), min(female_list))
max_height = max(max(male_list), max(female_list))

print("\n\n")
print("Partial Data")
male_height_bins = compute_bin(male_list)
female_height_bins = compute_bin(female_list)
print(female_height_bins)
print(male_height_bins)

print("\nMean Male Heights:")
print(get_mean(female_list))
print(get_mean(male_list))

print("\nMean Std Deviation:")
print(get_std_deviation(female_list))
print(get_std_deviation(male_list))

print("\nHistogram Result:")
print(get_female_height_probability(male_height_bins, female_height_bins,
                                    query_height))

print("\nSample Size:")
print(len(female_list))
print(len(male_list))

print("\nBaysian:")
print(get_female_height_prob_pdf(len(female_list), get_std_deviation(
    female_list), get_mean(female_list), len(male_list), get_std_deviation(
    male_list), get_mean(male_list), query_height))



