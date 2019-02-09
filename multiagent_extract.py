import os, pdb
import numpy as np

path = 'i101_trajectories-0750am-0805am.txt'

id2span = {}
with open(path, 'r') as f:
	for raw_data in f:
		data = raw_data.split()
		vid = data[0]
		if vid not in id2span:
			print(vid)
			id2span[vid] = [int(data[1]), int(data[1]) + int(data[2])]

max_t = 0
for span in id2span.values():
	s, d = span
	if d > max_t:
		max_t = d

spanid = {}
for pid, span in id2span.items():
	s, d = span
	for i in range(s, d):
		if i in spanid:
			spanid[i].append(int(pid))
		else:
			spanid[i] = [int(pid)]

# pdb.set_trace()
# with open('spanid', 'w+') as f:

# 	for t, ids in spanid.items():
# 		s = ''
# 		ids.sort()
# 		for pid in ids:
# 			s += str(pid) + '\t'
# 		f.write(str(t) + '\t')
# 		f.write(s + '\n')

agents = [2, 5, 8, 9, 10, 12, 13, 14, 20, 21, 22, 23, 25, 26, 27, 31, 32, 34, 37, 39, 40, 43]
start = []
end = []
for pid in agents:
	s, d = id2span[str(pid)]
	start.append(s)
	end.append(d)
	print(id2span[str(pid)])
pdb.set_trace()
start = max(start)
end = min(end)

# frame start-end 8-571
# 22 agents period 120 - 450
with open(path, 'r') as f:
	with open('i101_22agents-0750am-0800am.txt', 'w+') as g:
		for raw_data in f:
			data = raw_data.split()
			vid = data[0]
			if int(vid) in agents:
				g.write(raw_data)

np.save('id2span', id2span)