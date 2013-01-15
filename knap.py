#!/usr/bin/python

import sys, itertools, time, operator, copy, math, numpy

input_variable = int(sys.argv[2])

#load:
f = open(sys.argv[1])

instances = 0

a = {}
for line in f.readlines():
	sp = line.split()
	tmp_weights = []
	tmp_costs = []
	for i in xrange(len(sp[3:])/2):
		tmp_weights.append(int(sp[3 + i*2]))
		tmp_costs.append(int(sp[3 + i*2 + 1]))
	n = int(sp[1])
	M = int(sp[2])
	a[int(sp[0])] = {'n':int(sp[1]), 'M':int(sp[2]), 'weights':tmp_weights, 'costs':tmp_costs}
	instances = int(sp[1])

f.close()

data = a

def zeros_matrix(rows,cols):
	row = []
	data = []
	for i in range(cols):
		row.append(0)
	for i in range(rows):
		data.append(row[:])
	return data

def zeros_vector(cols):
	data = []
	for i in range(cols):
		data.append(0)
	return data

def matrix_pretty_print(matrix):
	for i in xrange(len(matrix) - 1, -1,  -1):
		print(str(i) + ':'), 
		for j in xrange(len(matrix[0])):
			print(matrix[i][j]),
		print ''

# ------------------------------- GENETIC -------------------------------------

class Member:
	data = []
	fitness = 0.0
	def __init__(self):
		self.data = list(numpy.random.randint(2, size=n))
	def mutate(self):
		random = numpy.random.randint(len(self.data))
		if (self.data[random] == 0):
			self.data[random] = 1
		else:
			self.data[random] = 0
	def mate(self, m):
		random = numpy.random.randint(len(m.data))
		return m.data[:random] + self.data[random:]
	def compute_fitness(self, weights, costs, M):
		fitness = 0.0
		weight = 0
		for i in xrange(len(self.data)):
			if (self.data[i]):
				weight += weights[i]
				fitness += costs[i]
		if (weight > M):
			fitness = 0.0
		self.fitness = fitness
		return fitness
	def __eq__(self, other):
		return self.fitness == other.fitness
	def __gt__(self, other):
		return self.fitness > other.fitness
	def __le__(self, other):
		return self.fitness < other.fitness
	def __str__(self):
		return str(self.fitness) + ':' + str(self.data) + '\n'
	def __repr__(self):
		return self.__str__()
	def get_fitness(self):
		return self.fitness
	def get_data(self):
		return self.data

DIVIDER = 5.0

def tournament(members, size):
	best_fitness = 0.0
	indices = numpy.random.randint(len(members), size=size)
#	indices = numpy.random.exponential(len(members) / DIVIDER, size=size)
	survivor = None
	for i in indices:
		index = int(i) % len(members)
		if members[index].get_fitness() > best_fitness:
			survivor = members[index]
			best_fitness = members[index].get_fitness()
	return survivor

def evolution_step(members, costs, weights, M,tournament_size, generation_size, mutation_chance):
	#hold a tournament, discard the old generation
	winners = []
	champion = None
	for i in xrange(generation_size):
		winner = tournament(members, tournament_size)
		assert(winner)
		winners.append(winner)

	mated = []
	successfull = 0
	elite_size = 0
	while successfull < generation_size - elite_size:
#		index1 = int(numpy.random.exponential(generation_size / DIVIDER)) % generation_size
#		index2 = int(numpy.random.exponential(generation_size / DIVIDER)) % generation_size
		index1 = numpy.random.randint(generation_size)
		index2 = numpy.random.randint(generation_size)
		assert(winners[index1].fitness)
		assert(winners[index2].fitness)
		child = winners[index1].mate(winners[index2])
		tmp = Member()
		tmp.data = child
		throw = numpy.random.randint(100)
		if (throw <= mutation_chance):
			tmp.mutate()
		if not(tmp.compute_fitness(weights, costs, M) == 0):
			successfull += 1
			mated.append(tmp)
		else:
			if not (mated == []):
				mated.append(mated[numpy.random.randint(successfull)])
				successfull += 1
	mated = mated + members[:elite_size]
	#Sort not really needed anymore
	mated.sort(reverse=True)
	
	return mated, mated[0].fitness, mated[0]

def genetic(costs, weights, M, tournament_size, generation_size, mutation_chance, iterations, acc_cost):
	alpha = None
	best_fitness = 0.0
	new_generation = []
	for i in xrange(generation_size):
		new_generation.append(Member())
		new_generation[i].compute_fitness(weights, costs, M)
	run = 0
	solution = None
	while run <= iterations:
		new_generation, fitness, champion = evolution_step(new_generation, costs, weights, M, tournament_size, generation_size, mutation_chance * 100)
		if fitness > best_fitness:
			best_fitness = fitness
			solution = champion
		print run, ((acc_cost - fitness) / float(acc_cost)) * 100
		run += 1
	return [solution.data, best_fitness]
	

# ------------------------------- END GENETIC ---------------------------------

inf = sys.maxint

def dynamic(costs_in, weights, M, epsilon):
	max_c = max(costs_in)
	if not (epsilon == 0):
		b = int(math.ceil((math.log((epsilon * max_c) / len(costs_in), 2))))
	else:
		b = 0
	costs = []
	for cost in costs_in:
		costs.append(cost >> b)
#		print 'cost: ', cost, 'shift: ', cost >> b
	total_sum = 0
	new_cost = 0
	# just follow what was on EDUX
	for cost in costs:
		total_sum += cost
	cost_table = zeros_matrix(total_sum + 1, len(costs) + 1)
	for j in (xrange(1, total_sum + 1)):
		cost_table[j][0] = inf
	for i in xrange(1, len(costs) + 1):
		for j in xrange(total_sum + 1):
#			print i, j
#			print cost_table[j][i - 1], cost_table[j - costs[i]][i - 1] + weights[i]
			cost_table[j][i] = min(cost_table[j][i - 1], cost_table[j - costs[i - 1]][i - 1] + weights[i - 1])
#	matrix_pretty_print(cost_table)
	out = zeros_vector(len(costs))

	j = 0
	solution_val = 0
	for j in xrange(total_sum, 0, -1):
		if cost_table[j][len(costs)] <= M:
			solution_val = j
			break
	j = solution_val
	for i in xrange(len(costs), 0, -1):
		if cost_table[j][i] != cost_table[j][i - 1]:
			out[i - 1] = 1
			j -= costs[i - 1]
#	print out, solution_val
	return [out, solution_val]

acc_solutions = {}

for key in data.iterkeys():
	res = dynamic(data[key]['costs'], data[key]['weights'], data[key]['M'], 0)
	acc_solutions[key] = res
heur_solutions = {}

heur_time = time.clock()
for key in data.iterkeys():
	res = genetic(data[key]['costs'], data[key]['weights'], data[key]['M'], 10, input_variable, 1.0, 50, acc_solutions[key][1])
#	res = knapsack_simple_heur(data[key]['costs'], data[key]['weights'], data[key]['n'], data[key]['M'])
	heur_solutions[key] = res
heur_time = time.clock() - heur_time
#print heur_time / len(data)

errortotal = 0.0
max_error = 0.0
max_key = 0
	
for key in data.iterkeys():
	brute = acc_solutions[key]
	approx = heur_solutions[key]
	error = (brute[1] - approx[1]) / float(brute[1])
	errortotal += error
	if error > max_error:
		max_error = error
		max_key = key
#print 'error:', (errortotal / len(data)) * 100, max_error * 100, max_key

