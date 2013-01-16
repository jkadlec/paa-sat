#!/usr/bin/python
import numpy, sys, time, matplotlib.pyplot as plt

class Util:
	def test_index(self, configuration, val):
		assert val != 0
		if val > 0:
			return configuration[val - 1]
		else:
			return not configuration[abs(val) - 1]
	def satisfied(self, configuration, expression):
		sat_count = 0
		for clause in expression:
			for log_val in clause:
				term = False
				if self.test_index(configuration, log_val):
					term = True
					break
			if term:
				sat_count += 1
		return sat_count

util = Util()

class Member:
	solver = None
	def __init__(self, solver, init):
		self.solver = solver
		if init:
			self.data = list(numpy.random.randint(2, size=solver.variables))
	def mutate(self):
		random = numpy.random.randint(len(self.data))
		if (self.data[random] == 0):
			self.data[random] = 1
		else:
			self.data[random] = 0
	def mate(self, m):
		random = numpy.random.randint(len(m.data))
		return m.data[:random] + self.data[random:]
	def compute_fitness(self):
		fitness = 0.0
		#compute all the clauses, if all are satisfied, boost the fitness massivelly
		sat_count = util.satisfied(self.data, solver.current_expression)
		if sat_count == solver.clauses:
			fitness += self.solver.ALL_SAT_BOOST
		else:
			fitness += sat_count * self.solver.SAT_BOOST
		#count the weights and be done with it
		self.cost = 0
		for i in xrange(len(self.data)):
			if self.data[i]:
				self.cost += solver.weights[i]
		fitness += self.cost * self.solver.WEIGHT_BOOST
		self.fitness = fitness
		return fitness
	def __eq__(self, other):
		return self.fitness == other.fitness
	def __gt__(self, other):
		return self.fitness > other.fitness
	def __le__(self, other):
		return self.fitness < other.fitness
	def __str__(self):
		return str(self.fitness) + ':' + str(self.data) + " " + str(self.cost) + '\n'
	def __repr__(self):
		return self.__str__()
	def get_fitness(self):
		return self.fitness
	def get_data(self):
		return self.data

DIVIDER = 5.0

class SAT_solver:
	def __init__(self, tournament_size, generation_size, mutation_probability, iterations, plot_errors, filename):
		self.filename = filename
		self.solved = 0
		self.tournament_size = tournament_size
		self.generation_size = generation_size
		self.mutation_probability = mutation_probability * 100
		self.iterations = iterations
		self.plot_errors = plot_errors
	def load_input(self):
		self.loaded, self.data = self.load_input_from_file()
	def solve_all(self):
		for item in self.data:
			item['found_solution'], item['found_fitness'] = self.solve_one(item)
	def load_input_from_file(self):
		f = open(self.filename, 'r')
		data_input = []
		loaded = 0
		for line in f.readlines():
			splitted = line.split();
			if 'c Known solution cost is' in line:
				best_cost = int(splitted[5])
				best_cost = 0
				if loaded:
					for i in xrange(len(best_solution)):
						if best_solution[i] == '1':
							best_cost += weights[i]
					data_input.append({'variables':variables, 'clauses':clauses,
                                                          'best_cost':best_cost, 'best_solution':best_solution,
                                                          'data':data, 'found_cost':0.0, 'weights':weights,
                                                          'found_solution':[], 'found_fitness':0})
			if 'c solution =' in line:
				best_solution = list(splitted[3])
			if splitted[0] == 'c':
				continue
			if splitted[0] == 's':
				best_cost = int(splitted[1])
				best_solution = list(splitted[2])
			elif splitted[0] == 'p':
				loaded += 1
				assert splitted[1] == 'cnf'
				variables = int(splitted[2])
				clauses = int(splitted[3])
				data = []
			elif splitted[0] == 'w':
				weights = []
				for i in xrange(variables):
					weights.append(int(splitted[i + 1]))
			else:
				v = []
				i = 0
				while splitted[i] != '0':
					v.append(int(splitted[i]))
					i += 1
				data.append(v)
		best_cost = 0
		for i in xrange(len(best_solution)):
			if best_solution[i] == '1':
				best_cost += weights[i]
		data_input.append({'variables':variables, 'clauses':clauses, 'best_cost':best_cost,
                                  'best_solution':best_solution, 'data':data, 'found_cost':0.0,
                                  'weights':weights, 'found_solution':[], 'found_fitness':0})
		print 'Loaded', loaded, 'instances.'
		f.close()
		self.histories = [[]] * loaded
		return loaded, data_input
	def write_solutions(self):
		f = open(self.filename+'sol', 'w')
		for item in self.data:
			f.write('c solution ' + item['found_solution'] + '\n')
			f.write('c found cost ' + item['found_cost'] + '\n')
			f.write('c cnf ' + str(item['variables']) + ' ' + str(item['clauses']) + '\n')
			for clause in item['data']:
				f.write(clause + ' 0' + '\n')
		f.close()
	def tournament(self, members, size):
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
	def evolution_step(self, members, tournament_size, generation_size, mutation_chance):
		#hold a tournament, discard the old generation
		winners = []
		champion = None
		for i in xrange(generation_size):
			winner = self.tournament(members, tournament_size)
			assert winner
			winners.append(winner)

		mated = []
		successfull = 0
		elite_size = 0
		while len(mated) < generation_size:
			index1 = numpy.random.randint(generation_size)
			index2 = numpy.random.randint(generation_size)
			child = winners[index1].mate(winners[index2])
			tmp = Member(self, False)
			tmp.data = child
			throw = numpy.random.randint(100)
			if (throw <= mutation_chance):
				tmp.mutate()
			mated.append(tmp)
			tmp.compute_fitness()
		mated = mated + members[:elite_size]
		mated.sort(reverse=True)
	
		return mated, mated[0].fitness, mated[0]
	def solve_one(self, input_data):
		alpha = None
		best_fitness = 0.0
		new_generation = []
		self.current_expression = input_data['data']
		self.clauses = input_data['clauses']
		self.variables = input_data['variables']
		self.weights = input_data['weights']

		self.SAT_BOOST = 10 * self.variables
		self.ALL_SAT_BOOST = 2 * self.variables
		self.WEIGHT_BOOST = 0#1 / 10 * self.variables

		start_time = time.clock()
	
		for i in xrange(self.generation_size):
			new_generation.append(Member(self, True))
			new_generation[i].compute_fitness()
		run = 0
		solution = None
		while run <= self.iterations:
			new_generation, fitness, champion = self.evolution_step(new_generation,
	                                                                        self.tournament_size,
	                                                                        self.generation_size,
	                                                                        self.mutation_probability)
			if fitness > best_fitness:
				best_fitness = fitness
				solution = champion
			self.histories[self.solved].append(solution)
			print run, fitness, solution, solution.cost, input_data['best_solution'], input_data['best_cost']
			run += 1
		self.solved += 1
		end_time = time.clock() - start_time
		return solution, best_fitness
	def create_plots(self):
		#plot only from first solution
		to_plot = self.histories[0]
		fitnesses = []
		costs = []
		for generation in to_plot:
			fitnesses.append(generation.get_fitness())
			costs.append(generation.cost)
		plt.plot(fitnesses)
		plt.show()
		plt.plot(costs)
		plt.show()
		if self.plot_errors:
			assert False
			
		

solver = SAT_solver(20, 100, 0.9, 300, False, sys.argv[1])
solver.load_input()
solver.solve_all()
solver.create_plots()

#genetic(input_data[0], 20, 100, 0.9, 300)
