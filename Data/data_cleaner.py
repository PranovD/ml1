#input: data from communties_data.txt
#output: data useful for predictive analysis (ie: excluding first 5 attributes) in 2D Array
def clean_raw_data(data):
	rows = (row.strip().split() for row in data)
	leaveLoop = 0
	cleaned_data = []
	for row in rows:
		tmp = [stat.strip() for stat in row[0].split(',')]
		cleaned_data.append(tmp[5:])
	return cleaned_data
#input: data from summary.txt
#output: data in 2D Array
def clean_sum_data(data):
	# Summary data contains the Attribute, Min, Max, Mean, SD, Correl, Median, Mode, Missing
	sum_rows = (row.strip().split() for row in data)
	new_list = []
	for row in sum_rows:
	 	new_list.append(row)
	return new_list

#input: output of clean_raw_data, clean_sum_data, use median value
#output clean_raw_data with the '?' values replaced with the median value if True or mean if False
def replace_null_data(old_replacee, replacer, median=True):
	replacee = list(old_replacee)
	for row in replacee:
		replaced = False
		old_row = list(row)
		for col in range(len(row)):
			if row[col] == '?':
				replaced = True
				if median:
					row[col] = replacer[col][6]
				else:
					row[col] = replacer[col][3]
	return replacee

# Looks funny but this makes sure it runs when called by a python file in a different directory and still work when run locally
summary_data = open("../Data/summary.txt", "r")
raw_data = open("../Data/communities_data.txt", "r")

cleaned_data = clean_raw_data(raw_data)
sumarized_data = clean_sum_data(summary_data)
usable_data = replace_null_data(cleaned_data, sumarized_data)