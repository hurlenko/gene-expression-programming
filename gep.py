import numpy as np


def fitness(x):
    # return x ** 2
    # return (x + x * x) / (x ** 3)
    return (x * x) + (x * x) - (x + x)


class GEP(object):
    def __init__(self, grammar, bounds, len_head, iterations, pop_size, prob_crossover, num_trials):
        self.grammar = grammar
        self.bounds = bounds
        self.len_head = len_head
        self.len_tail = len_head * (len(grammar['TERM'])) + 1
        self.iterations = iterations
        self.pop_size = pop_size
        self.prob_crossover = prob_crossover
        self.num_trials = num_trials
        self.population = [Individual(self.random_genome()) for _ in range(self.pop_size)]
        self.best = sorted(self.population, key=lambda x: x.fitness)[0]

    def binary_tournament(self):
        a, b = np.random.choice(self.population, 2)
        return a if a.fitness < b.fitness else b

    def point_mutation(self, genome):
        rate = 1.0 / len(genome)
        child = list()
        for i in range(len(genome)):
            bit = genome[i]
            if np.random.random() < rate:
                if i < self.len_head:
                    selection = self.grammar['FUNC'] if np.random.random() < 0.5 else self.grammar['TERM']
                    bit = selection[np.random.randint(0, len(selection))]
                else:
                    bit = self.grammar['TERM'][np.random.randint(0, len(self.grammar['TERM']))]
            child.append(bit)
        return ''.join(child)

    def crossover(self, parent1, parent2):
        if np.random.random() < self.prob_crossover:
            return ''.join([a if np.random.random() < 0.5 else b for a, b in zip(parent1, parent2)])
        else:
            return parent1

    def reproduce(self, selected):
        children = list()
        for a, b in zip(selected[::2], selected[1::2]):
            children.append(Individual(self.point_mutation(self.crossover(a.genome, b.genome))))
        return children

    def random_genome(self):
        s = list()
        for _ in range(self.len_head):
            selection = self.grammar['FUNC'] if np.random.random() < 0.5 else self.grammar['TERM']
            s.append(selection[np.random.randint(0, len(selection))])
        for _ in range(self.len_tail):
            s.append(self.grammar['TERM'][np.random.randint(0, len(self.grammar['TERM']))])
        return ''.join(s)

    @staticmethod
    def cost(program):
        errors = 0.0
        if FROM_TABLE:
            for x, y in TABLE:
                expression = program.replace('x', str(x))
                try:
                    score = eval(expression)
                except ZeroDivisionError:
                    score = float('inf')
                errors += abs(score - y)
            if np.isnan(errors):
                errors = float('inf')
            return errors / len(TABLE)
        else:
            for _ in range(NUM_TRIALS):
                x = BOUNDS[0] + ((BOUNDS[1] - BOUNDS[0]) * np.random.random())
                expression = program.replace('x', str(x))
                try:
                    score = eval(expression)
                except ZeroDivisionError:
                    score = float('inf')
                errors += np.abs(score - fitness(x))
            return errors / NUM_TRIALS

    @staticmethod
    def mapping(genome):
        off = 0
        queue = list()
        root = {'node': genome[off]}
        off += 1
        queue.append(root)
        while len(queue) > 0:
            current = queue.pop(0)
            if current['node'] in GRAMMAR['FUNC']:
                current['left'] = {'node': genome[off]}
                off += 1
                queue.append(current['left'])
                current['right'] = {'node': genome[off]}
                off += 1
                queue.append(current['right'])
        return root

    @staticmethod
    def tree_to_string(exp):
        if 'left' not in exp or 'right' not in exp:
            return exp['node']
        left = GEP.tree_to_string(exp['left'])
        right = GEP.tree_to_string(exp['right'])
        if exp['node'] in ['+', '-', '*', '/']:
            return '({0} {1} {2})'.format(left, exp['node'], right)
        else:
            func = {
                'P': 'np.power',
            }.get(exp['node'], None)
            return '({0}({1}, {2}))'.format(func, left, right)

    def run(self):
        for gen in range(1, self.iterations + 1):
            selected = [self.binary_tournament() for _ in range(self.pop_size)]
            children = self.reproduce(selected)
            children = list(sorted(children, key=lambda x: x.fitness))
            self.best = children[0] if children[0].fitness <= self.best.fitness else self.best
            self.population = sorted((children + self.population), key=lambda x: x.fitness)[:self.pop_size]
            if gen % 50 == 0:
                print('{0}/{1} Current population:'.format(gen, self.iterations))
                print(self.best)
            if self.best.fitness == 0.0:
                print('{0}/{1} Current population:'.format(gen, self.iterations))
                print(self.best)
                break
        return self.best


class Individual(object):
    def __init__(self, genome):
        self.genome = genome
        self.expression = GEP.mapping(genome)
        self.program = GEP.tree_to_string(self.expression)
        self.fitness = GEP.cost(self.program)

    def __str__(self):
        return '{0} = {1}'.format(self.program, self.fitness)


GRAMMAR = {'FUNC': ['+', '-', '*', '/'], 'TERM': ['x']}
BOUNDS = [1.0, 10.0]
TABLE = [
    [0, 0],
    [1, 1],
    [2, 4],
    [3, 9],
    [4, 16],
    [5, 25],
    [6, 36],
    [7, 49],
    [8, 64],
    [9, 81],
    [10, 100]
]
NUM_TRIALS = 30
FROM_TABLE = False
LEN_HEAD = 30
ITERATIONS = 200
POP_SIZE = 100
PROB_CROSS = 0.85


def main():
    gep = GEP(GRAMMAR, BOUNDS, LEN_HEAD, ITERATIONS, POP_SIZE, PROB_CROSS, NUM_TRIALS)
    gep.run()

if __name__ == '__main__':
    main()
