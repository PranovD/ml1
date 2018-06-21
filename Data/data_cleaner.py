# Summary data contains the Min, Max, Mean, SD, Correl, Median, Mode, Missing
summary_data = open("summary.txt", "r")
sum_rows = (row.strip().split() for row in summary_data)
new_list = []
for row in sum_rows:
 	new_list.append(row)
sum_rows = new_list
raw_data = open("communities_data.txt", "r")
rows = (row.strip().split() for row in raw_data)
leaveLoop = 0
clean_data = []
for row in rows:
#	print "This is the row"
#	print row
#	print "This is the row[0]"
#	print row[0]
	tmp = [stat.strip() for stat in row[0].split(',')]
#	print tmp
	clean_data.append(tmp[5:])
#print clean_data
print len(clean_data[0])
# print sum_rows
print len(sum_rows)
for row in clean_data:
	replaced = False
	old_row = list(row)
	for col in range(len(row)):
		if row[col] == '?':
			replaced = True
			row[col] = sum_rows[col][6]
	if replaced:
		print 'old_row'
		print old_row
		print 'row'
		print row
