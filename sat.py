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
	def satisfiable_by(self, configuration, expression, index):
		sat_count = 0
		val = configuration[index]
		if val == 0:
			for clause in expression:
				if -(index + 1) in clause:
					sat_count += 1
		else:
			for clause in expression:
				if (index + 1) in clause:
					sat_count += 1
		return sat_count
			
util = Util()

class Member:
	solver = None
	def __init__(self, solver, init):
		self.solver = solver
		self.satisfied = False
		if init:
			self.data = [ 0 ] * solver.variables
#			self.data = list(numpy.random.randint(2, size=solver.variables))
		
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
			fitness += sat_count * self.solver.SAT_BOOST
			self.satisfied = True
		else:
			fitness += sat_count * self.solver.SAT_BOOST
			self.satisfied = False
		self.sat_count = sat_count
		#count the weights and be done with it
		self.cost = 0
		for i in xrange(len(self.data)):
			if self.data[i]:
				self.cost += solver.weights[i]
		fitness += self.cost * self.solver.WEIGHT_BOOST
		variable_score = 0
		for i in xrange(len(self.data)):
			variable_score += util.satisfiable_by(self.data, solver.current_expression, i)
		fitness += variable_score * self.solver.VARIABLE_BOOST
		self.variable_score = variable_score
		self.fitness = fitness
		return fitness
	def __eq__(self, other):
		return self.fitness == other.fitness
	def __gt__(self, other):
		return self.fitness > other.fitness
	def __le__(self, other):
		return self.fitness < other.fitness
	def __str__(self):
		return str(self.fitness) + ':' + str(self.data) + " " + str(self.cost) + " " + str(self.satisfied) + '\n'
	def __repr__(self):
		return self.__str__()
	def get_fitness(self):
		return self.fitness
	def get_data(self):
		return self.data

DIVIDER = 5.0

class SAT_solver:
	def __init__(self, tournament_size, generation_size, mutation_probability, iterations, sat_weight, all_sat_weight, weight_weight, variable_weight, plot_errors, filename):
		self.filename = filename
		self.solved = 0
		self.tournament_size = tournament_size
		self.generation_size = generation_size
		self.mutation_probability = mutation_probability * 100
		self.iterations = iterations
		self.plot_errors = plot_errors
		self.sat_weight = sat_weight
		self.all_sat_weight = all_sat_weight
		self.weight_weight = weight_weight
		self.variable_weight = variable_weight
	def pretty_print_input(self, func_input):
		for bracket in func_input:
			print '(',
			for var in bracket:
				if var > 0:
					print 'x_'+str(var + 1)+' +',
				else:
					print 'x_'+str(-var + 1)+'\' +',
			print ') * ',
		print
	def load_input(self):
		self.loaded, self.data = self.load_input_from_file()
	def solve_all(self):
		for item in self.data:
			item['found_solution'], item['found_fitness'] = self.solve_one(item)
			print item['found_solution'], item['best_solution'], item['best_cost']
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
		self.histories = []
		for i in xrange(loaded):
			self.histories.append({'time':0.0, 'data':[], 'first_sat':None})
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
	def solve_one(self, input_data, variable):
		alpha = None
		best_fitness = 0.0
		new_generation = []
		self.current_expression = input_data['data']
#		self.pretty_print_input(input_data['data'])
		self.clauses = input_data['clauses']
		self.variables = input_data['variables']
		self.weights = input_data['weights']

		weight_sum = sum(self.weights)

		self.SAT_BOOST = self.sat_weight# * self.variables
		self.ALL_SAT_BOOST = self.all_sat_weight * weight_sum# * self.variables
		self.WEIGHT_BOOST = self.weight_weight #1 / 10 * self.variables
		self.VARIABLE_BOOST = self.variable_weight

		start_time = time.clock()
	
		for i in xrange(self.generation_size):
			new_generation.append(Member(self, True))
			new_generation[i].compute_fitness()
		run = 0
		solution = None
		assert self.histories[self.solved]['data'] == []
		first_sat = None
		while run < self.iterations:
			new_generation, fitness, champion = self.evolution_step(new_generation,
	                                                                        self.tournament_size,
	                                                                        self.generation_size,
	                                                                        self.mutation_probability)
			if fitness > best_fitness:
				best_fitness = fitness
				solution = champion
			if first_sat == None and champion.satisfied:
				first_sat = run
				self.histories[self.solved]['first_sat'] = run
			self.histories[self.solved]['data'].append(solution)
#			print run, fitness, solution, solution.cost, input_data['best_solution'], input_data['best_cost']
			run += 1
		assert len(self.histories[self.solved]['data']) == self.iterations
		end_time = time.clock() - start_time
		self.histories[self.solved]['time'] = end_time
		self.solved += 1
		return solution, best_fitness
	def create_plots_one(self):
		#plot only from first solution
		to_plot = self.histories[0]
		assert len(to_plot['data']) == self.iterations
		fitnesses = []
		costs = []
		sat = []
		for generation in to_plot['data']:
			fitnesses.append(generation.get_fitness())
			costs.append(generation.cost)
			if generation.satisfied:
				sat.append(1)
			else:
				sat.append(0)
		plt.plot(fitnesses, lw=0.75, color='r')
		plt.xlabel('Pocet iteraci')
		plt.ylabel('Hodnota fitness funkce nejlepsiho jedince')
		plt.title('Zavislost hodnoty fitness funkce na poctu iteraci')
		if (to_plot['first_sat'] != None):
			x = to_plot['first_sat']
			y = to_plot['data'][to_plot['first_sat']].get_fitness()
			plt.annotate('F(Y) = 1', xy=(x,y), xytext=(x + (self.iterations / 100.0) * 3, y - ((max(fitnesses) - min(fitnesses)) / 100.0) * 10), arrowprops=dict(facecolor='black', shrink=0.05))
		plt.xlabel('Pocet iteraci')
		plt.grid(True)
		plt.show()
		plt.plot(costs, lw=0.75, color='r')
		if (to_plot['first_sat'] != None):
			x = to_plot['first_sat']
			y = to_plot['data'][to_plot['first_sat']].cost
			plt.annotate('F(Y) = 1', xy=(x,y), xytext=(x + (self.iterations / 100.0) * 3, y - ((max(costs) - min(costs)) / 100.0) * 10), arrowprops=dict(facecolor='black', shrink=0.05))
		plt.xlabel('Pocet iteraci')
		plt.ylabel('Hodnota ceny nejlepsiho jedince')
		plt.title('Zavislost hodnoty ceny na poctu iteraci')
		plt.grid(True)
		plt.show()
		plt.plot(sat, lw=0.75, color='r')
		plt.title('')
		plt.grid(True)
		plt.show()
		#times for all instances
		mean_time = 0.0
		for hist in self.histories:
			mean_time += hist['time']
		print 'average time for 1 instance', mean_time / float(self.solved)
		print 'F() solved', to_plot['first_sat']
	def create_plots_one(self):
			
def sat_test():
	data = [[5, -2, 6], [-6, 5, -7], [-3, -2, -1]]
	assert util.satisfied([0, 0, 0, 0, 1, 0, 0], data) == 3
	assert util.satisfied([0, 0, 0, 0, 0, 1, 1], data) == 2

SAT_WEIGHT = 30
ALL_SAT_WEIGHT = 1.2
WEIGHT_WEIGHT = 1
VARIABLE_WEIGHT = 50

do_test = True


tournament_sizes = []

sat_test()
solver = SAT_solver(25, 50, 1, 31, SAT_WEIGHT, ALL_SAT_WEIGHT, WEIGHT_WEIGHT, VARIABLE_WEIGHT, False, sys.argv[1])
solver.load_input()
solver.solve_all()
solver.create_plots()

#genetic(input_data[0], 20, 100, 0.9, 300)
