import sys, copy

DBG = 0

sys.setrecursionlimit(100000)

def debug(*msg):
	if DBG == 1:
		print msg

def expand(q, closed, state, instances, capacities):
	expanded_states = []
	moves = []
	#try to apply each move and see if it's relevant
	#there is a fixed number of actions to be done with each bucket
	for i in xrange(instances):
		#try to move water from this bucket to all other buckets
		for j in xrange(instances):
			#i've got i - moving from and j - moving to - does not apply in three cases - i == 0 and j == full and i == j
			if (state[i] == 0 or state[j] == capacities[j] or i == j):
				continue
			else:
				debug('Trying to move water from bucket ', i, 'to bucket ', j)
				modified = copy.deepcopy(state)
				modified[i] = state[i] - (min(state[i], capacities[j] - state[j]))
				modified[j] = state[j] + (min(state[i], capacities[j] - state[j]))
				#append operation
				modified.append('move' + str(i) + str(j))
				if not (modified in q or modified in closed):
					expanded_states.append(modified)
	#			expanded_states.append(state[0:max(0, min(i,j) - 1)] + [state[min(i,j)] - state[max(i,j)] + state[max(i,j):])
		#try to fill this bucket
		#does this make sense if bucket is not empty but not full? I THINK NOT (check!!!)
		if state[i] == 0:
			#this fills this bucket and append the newly created state
#			expanded_states.append(state[0:max(0,i - 1) + [capacities[i]] + state[i:]])
			modified = copy.deepcopy(state)
			modified[i] = capacities[i]
			modified.append('fill' + str(i))
			if not (modified in q or modified in closed):
				expanded_states.append(modified)
		#try to empty the bucket, do not empty empty buckets or full buckets
		if state[i] != 0 and state[i] != capacities[i]:
#			expanded_states.append(state[0:max(0,i - 1) + [0] + state[i:]])
			modified = copy.deepcopy(state)
			modified[i] = 0
			modified.append('empty' + str(i))
			if not (modified in q or modified in closed):
				expanded_states.append(modified)

	return expanded_states, moves
		

def bfs(state, solution, moves, q, instances, capacities, depth, closed):
	debug('BFS: q:', q)
	#do we have a solution?
	if state[0:instances] == solution:
		print 'solution found in depth', depth, 'q length: ', len(q), 'tried states: ', len(closed)
		return True
	if q == []:
		print 'no solution'
		return False
	#it's not a solution
	closed.append(state)
	#expand the state, this is a tuple with states and according moves to obtain them
	ex = expand(q, closed, state, instances, capacities)
	debug('BFS: expanded following states', ex[0], 'and moves', ex[1])
	#enque the state
	state = q[0]
	debug('removing state', state, 'from queue')
	#remove state
	q = q[1:]
	q = q + ex[0]
	#recurse
	bfs(state[0:instances], solution, ex[1], q, instances, capacities, depth + 1, closed)
	
f = open(sys.argv[1], 'r')

data = {}
for line in f.readlines():
	if line == '\n':
		break
	sp = line.split()
	bucket_count = int(sp[1])
	capacities = []
	for i in xrange(bucket_count):
		capacities.append(int(sp[i+2]))
	initial = []
	for i in xrange(bucket_count):
		initial.append(int(sp[i+2+bucket_count]))
	target = []
	for i in xrange(bucket_count):
		target.append(int(sp[i+2+(bucket_count * 2)]))
	data[int(sp[0])] = {'cap':capacities, 'init':initial, 'tar':target }

f.close()

#solution
moves = []
#state for each bucket - how much water is in it
state = []
for i in xrange(bucket_count):
	state.append(0)

#queue of waiting operations
queue = []

solutions = {}

for key in data.iterkeys():
	item = data[key]
	#set the first state
	state = item['init']
	moves = []
	bfs(state, item['tar'], moves, [state], bucket_count, item['cap'], 0, [])
	solutions[key] = moves

print solutions
