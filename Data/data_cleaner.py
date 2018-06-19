data = open("communities_data.txt", "r")
rows = (row.strip().split() for row in data)
leaveLoop = 0
for row in rows:
	leaveLoop += 1
	print row
	if leaveLoop > 10:
		break


