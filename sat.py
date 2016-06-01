#!/usr/bin/python
import numpy, sys, time, matplotlib.pyplot as plt, pdb

#SETTINGS

SAT_WEIGHT = 8
VARIABLE_WEIGHT = 2
WEIGHT_WEIGHT = 3
ALL_SAT_WEIGHT = 10 #has to be more than all variables set to 1
COUNT = None #how many input instances do we solve? If set to None, all will be solved
TEST = False #testing GA arguments
TEST_FITNESS = False #testing fitness function arguments

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
		return str(self.fitness) + ':' + str(self.data) + " " + str(self.cost) + " " + str(self.satisfied)
	def __repr__(self):
		return self.__str__()
	def get_fitness(self):
		return self.fitness
	def get_data(self):
		return self.data

DIVIDER = 5.0

class SAT_solver:
	def __init__(self, tournament_size, generation_size, mutation_probability, iterations, sat_weight, all_sat_weight, weight_weight, variable_weight, var_array, elite_size, filename):
		self.filename = filename
		self.solved = 0
		self.tournament_size = tournament_size
		self.generation_size = generation_size
		self.mutation_probability = mutation_probability * 100
		self.iterations = iterations
		self.sat_weight = sat_weight
		self.all_sat_weight = all_sat_weight
		self.weight_weight = weight_weight
		self.variable_weight = variable_weight
		self.var_array = var_array
		self.elite_size = elite_size
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
	def solve_all(self, variable, save_results, limit):
		self.solved = 0
		i = 0
		for item in self.data:
			print 'solving', i
			if TEST:
				item['found_solution'], item['found_cost'] = self.solve_one(item, variable)
			else:
				item['found_solution'], item['found_cost'] = self.solve_one(item, None)
			print self.solved, item['found_solution'].satisfied, item['found_solution'].cost
			i += 1
			if limit:
				if limit == i:
					break
		if save_results:
			self.write_solutions()
	def init_history(self, loaded):
		if TEST:
			self.histories = {}
			for var in self.var_array:
				tmp = []
				for i in xrange(loaded):
					tmp.append({'time':0.0, 'data':[], 'first_sat':None})
				self.histories[var] = tmp
		else:
			self.histories = []
			for i in xrange(loaded):
				self.histories.append({'time':0.0, 'data':list(), 'first_sat':None})
	def load_input_from_file(self):
		f = open(self.filename, 'r')
		data_input = []
		loaded = 0
		for line in f.readlines():
			splitted = line.split();
			if 'c by Jan Kadlec' in line:
#				best_cost = int(splitted[5])
#				best_cost = 0
				if loaded:
#					for i in xrange(len(best_solution)):
#						if best_solution[i] == '1':
#							best_cost += weights[i]
					data_input.append({'variables':variables, 'clauses':clauses,
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
				if loaded:
#					for i in xrange(len(best_solution)):
#						if best_solution[i] == '1':
#							best_cost += weights[i]
					data_input.append({'variables':variables, 'clauses':clauses,
                                                          'data':data, 'found_cost':0.0, 'weights':weights,
                                                          'found_solution':[], 'found_fitness':0})
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
#		for i in xrange(len(best_solution)):
#			if best_solution[i] == '1':
#				best_cost += weights[i]
		data_input.append({'variables':variables, 'clauses':clauses, 'best_cost':best_cost,
                                   'data':data, 'found_cost':0.0,
                                  'weights':weights, 'found_solution':[], 'found_fitness':0})
		print 'Loaded', loaded, 'instances.'
		self.init_history(loaded)
		f.close()
		return loaded, data_input
	def write_solutions(self):
		f = open(self.filename+'.sol', 'w')
		i = 0
		for item in self.data:
			if i == self.solved:
				break
#			if not item['found_solution'].satisfied:
#				continue
			f.write('c solution ' + str(item['found_solution'].data) + '\n')
			f.write('c found cost ' + str(item['found_cost']) + '\n')
			f.write('c cnf ' + str(item['variables']) + ' ' + str(item['clauses']) + '\n')
			for clause in item['data']:
				clause_str = ''
				for one in clause:
					clause_str += str(one) + ' '
				f.write(clause_str + '0' + '\n')
			i += 1
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
		while len(mated) < generation_size:
			index1 = numpy.random.randint(generation_size)
			index2 = numpy.random.randint(generation_size)
			child = winners[index1].mate(winners[index2])
			tmp = Member(self, False)
			tmp.data = child
			throw = numpy.random.randint(100)
			if (throw < mutation_chance):
				tmp.mutate()
			mated.append(tmp)
			tmp.compute_fitness()
		mated = mated + members[:self.elite_size]
		winner = max(mated)
		return mated, winner.fitness, winner
	def solve_one(self, input_data, variable):
		alpha = None
		best_fitness = 0.0
		new_generation = []
		self.current_expression = input_data['data']
#		self.pretty_print_input(input_data['data'])
		self.clauses = input_data['clauses']
		self.variables = input_data['variables']
		self.weights = input_data['weights']

		weight_sum_tmp = sum(self.weights)
		weight_sum = weight_sum_tmp
#		while weight_sum > 100:
#			weight_sum /= 10

		self.SAT_BOOST = self.sat_weight * (weight_sum / float(self.variables)) * self.clauses
		self.ALL_SAT_BOOST = self.all_sat_weight * (weight_sum / float(self.variables)) * self.clauses * self.variables
		self.WEIGHT_BOOST = self.weight_weight * self.clauses
		self.VARIABLE_BOOST = self.variable_weight * (weight_sum / float(self.variables))

		start_time = time.clock()
	
		for i in xrange(self.generation_size):
			new_generation.append(Member(self, True))
			new_generation[i].compute_fitness()
		run = 0
		solution = new_generation[0]
		first_sat = None
		while run < self.iterations:
			new_generation, fitness, champion = self.evolution_step(new_generation,
	                                                                        self.tournament_size,
	                                                                        self.generation_size,
	                                                                        self.mutation_probability)
			if fitness > best_fitness:
				if not (solution.satisfied and not champion.satisfied):
					best_fitness = fitness
					solution = champion
			if first_sat == None and champion.satisfied:
				first_sat = run
				if variable == None:
					self.histories[self.solved]['first_sat'] = run
				else:
					if self.histories.has_key(variable):
						self.histories[variable][self.solved]['first_sat'] = run
					else:
						assert False
			if variable == None:
				self.histories[self.solved]['data'].append(solution)
			else:
				self.histories[variable][self.solved]['data'].append(solution)
			run += 1
		end_time = time.clock() - start_time
		if variable == None:
			self.histories[self.solved]['time'] = end_time
		else:
			self.histories[variable][self.solved]['time'] = end_time
		self.solved += 1
		return solution, solution.cost
	def create_plots_one(self, index):
		if index != None:
			to_plot = self.histories[index]
		else:
			to_plot = self.histories[numpy.random.randint(self.solved)]

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
	def create_plots_var_test(self, variable_array, variable_string, legend):
		#plot only from first solution
		costs = {}
		fitnesses = {}
		for i in variable_array:
			costs[i] = []
			fitnesses[i] = []
		times = []
		rand_index = numpy.random.randint(self.solved)
		sat_dict = {}
		for var in variable_array:
			satisfied = 0
			for data in self.histories[var][rand_index]['data']:
				costs[var].append(data.cost)
				fitnesses[var].append(data.fitness)
			assert len(costs[var]) == self.iterations
			assert len(fitnesses[var]) == self.iterations
			mean_time = 0.0
			for data2 in self.histories[var]:
				mean_time += data2['time']
				if data2['first_sat']:
					satisfied += 1
			sat_dict[var] = satisfied
			times.append(mean_time / self.solved)
		plotargs_fit = []
		plotargs_cost = []
		for i in xrange(self.iterations):
			plotargs_fit.append([])
			plotargs_cost.append([])
			for j in xrange(len(variable_array)):
				plotargs_fit[i].append([])
				plotargs_cost[i].append([])
		for i in xrange(self.iterations):
			j = 0
			for var in variable_array:
				assert len(costs[var]) == self.iterations
				plotargs_cost[i][j] = (costs[var][i])
				plotargs_fit[i][j] = (fitnesses[var][i])
				j += 1
			assert j == len(variable_array)
		assert len(plotargs_fit[0]) == len(variable_array)
		plt.plot(plotargs_fit)
		plt.xticks(range(1, self.iterations + 1, 2), range(1, self.iterations + 1, 2))
		plt.xlabel('Pocet iteraci')
		plt.ylabel('Hodnota fitness funkce nejlepsiho jedince')
		plt.title('Vyvoj fitness funkce v case - promenna='+variable_string)
		plt.xlabel('Pocet iteraci')
		plt.grid(True)
		plt.legend(legend, loc='lower right', prop={'size':10})
		plt.savefig('fit.png', format='png', dpi=200)
		plt.show()
		plt.plot(plotargs_cost)
		plt.xlabel('Pocet iteraci')
		plt.ylabel('Hodnota ceny nejlepsiho jedince')
		plt.title('Vyvoj ceny v case - promenna='+variable_string)
		plt.grid(True)
		plt.xticks(range(1, self.iterations, 2), range(1, self.iterations, 2))
		plt.legend(legend, loc='lower right', prop={'size':10})
		plt.savefig('cena.png', format='png', dpi=200)
		plt.show()
		plt.plot(times, color='r')
		plt.grid(True)
		plt.xlabel(variable_string)
		plt.title('Casy na jednu instanci - promenna='+variable_string)
		plt.ylabel('Prumerny cas na reseni jedne instance')
		plt.xticks(range(len(variable_array)), variable_array)
		plt.ylim(ymin=0.0, ymax=max(times) + (max(times) / 10))
		plt.savefig('casy.png', format='png', dpi=200)
		plt.show()
		print '=============TIMES TABLE============='
		print '^', variable_string, '^ Cas na jednu instanci ^'
		i = 0
		for var in variable_array:
			print '|', var, '|', times[i], '|'
			i += 1
		print '=============SAT PERCENTAGE============='
		print '^', variable_string, '^ Procento splnenych instanci ^'
		for var in variable_array:
			print '|', var, '|', (sat_dict[var] / float(self.solved)) * 100, '% |'
			
def sat_test():
	data = [[5, -2, 6], [-6, 5, -7], [-3, -2, -1]]
	assert util.satisfied([0, 0, 0, 0, 1, 0, 0], data) == 3
	assert util.satisfied([0, 0, 0, 0, 0, 1, 1], data) == 2

sat_test()
if TEST:
	legend = []
	solver = SAT_solver(10, 80, 0.98, 75, SAT_WEIGHT, ALL_SAT_WEIGHT, WEIGHT_WEIGHT, VARIABLE_WEIGHT, probabilities, 0, sys.argv[1])
	solver.load_input()
#	TESTING PROBABILITY
#	probabilities = [95, 96, 97, 98, 99, 100]
#	for size in tournament_sizes:
#	for prob in probabilities:
#		solver.mutation_probability = prob
#		legend.append(str(prob) + '%')
#		solver.solve_all(prob, False, COUNT)
#	solver.create_plots_var_test(probabilities, "Pravdepodobnost mutace", legend)
#	legend = []
	solver.mutation_probability = 95
#	TESTING GENERATION SIZE
#	generation_sizes = [10, 20, 50, 75, 100, 150, 200, 250, 300]
#	solver.var_array = generation_sizes
#	solver.init_history(solver.loaded)
#	for size in generation_sizes:
#		solver.generation_size = size
#		legend.append(str(size))
#		solver.solve_all(size, False, COUNT)
#	solver.create_plots_var_test(generation_sizes, "Velikost generace", legend)
	solver.generation_size = 200
	solver.tournament_size = (1.0 / 50) * solver.generation_size
#	TESTING TOURNAMENT_SIZE
#	legend = []
#	tournament_sizes = [100, 50, 25, 20, 10, 5, 2, 3]
#	actual_sizes = [(1.0 / 100) * 300, (1.0 / 50) * 300, (1.0 / 25) * 300, (1.0 / 20) * 300, (1.0 /10) * 300, (1.0 / 5) * 300, (1.0 / 2) * 300, (1.0 / 3) * 300]
#	solver.var_array = actual_sizes
#	solver.init_history(solver.loaded)
#	i = 0
#	for size in actual_sizes:
#		solver.tournament_size = (1.0 / size) * solver.generation_size
#		solver.tournament_size = size
#		legend.append('1/'+str(int(tournament_sizes[i])))
#		tournament_sizes[i] = 1.0 / size * solver.generation_size
#		i += 1
#		solver.solve_all(size, False, COUNT)
#	solver.create_plots_var_test(actual_sizes, "Velikost turnaje / velikost generace", legend)
#	legend = []
#	TESTING ELITE SIZE
	elite_sizes = [0, 1, 2, 3, 5, 10, 25, 30]
	solver.var_array = elite_sizes
	solver.init_history(solver.loaded)
	for size in elite_sizes:
		solver.elite_size = size
		legend.append(str(size))
	solver.solve_all(size, False, COUNT)
	solver.create_plots_var_test(elite_sizes, "Velikost elitni skupiny", legend)

elif TEST_FITNESS:
	#test of fitness function
	solver = SAT_solver(10, 80, 0.99, 100, SAT_WEIGHT, ALL_SAT_WEIGHT, WEIGHT_WEIGHT, VARIABLE_WEIGHT, False, 0, sys.argv[1])
	solver.load_input()
	solver.solve_all(None, False, 10)
	#random plot
#	solver.create_plots_one(None)
	#plot all
	print 'plots for ' , solver.solved, 'instances'
	for i in xrange(solver.solved):
		solver.create_plots_one(i)
else:
	#normal operation
	solver = SAT_solver(1.0 / 50 * 200, 200, 1, 75, SAT_WEIGHT, ALL_SAT_WEIGHT, WEIGHT_WEIGHT, VARIABLE_WEIGHT, False, 1, sys.argv[1])
	solver.load_input()
	solver.solve_all(None, True, COUNT)

#genetic(input_data[0], 20, 100, 0.9, 300)
