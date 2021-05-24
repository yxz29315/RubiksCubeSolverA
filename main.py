import random
import math
import time
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import clone_model


class Cube2x2:

    def __init__(self, state=['B','B','B','B','O','O','O','O','R','R','R','R','Y','Y','Y','Y','W','W','W','W','G','G','G','G']):
        self.state = state.copy()

    def isSolved(self):

        face1 = self.state[:4]
        face2 = self.state[4:8]
        face3 = self.state[8:12]

        for color in face1:
            if color != face1[0]:
                return False
        for color in face2:
            if color != face2[0]:
                return False
        for color in face3:
            if color != face3[0]:
                return False

        return True

    def move(self, turn):

        temp_state = self.state.copy()

        if turn == 'F':
            self.state[0] = temp_state[2]
            self.state[1] = temp_state[0]
            self.state[3] = temp_state[1]
            self.state[2] = temp_state[3]
            self.state[14] = temp_state[7]
            self.state[15] = temp_state[5]
            self.state[8] = temp_state[14]
            self.state[10] = temp_state[15]
            self.state[17] = temp_state[8]
            self.state[16] = temp_state[10]
            self.state[7] = temp_state[17]
            self.state[5] = temp_state[16]

        elif turn == 'Fp':
            self.state[0] = temp_state[1]
            self.state[1] = temp_state[3]
            self.state[3] = temp_state[2]
            self.state[2] = temp_state[0]
            self.state[14] = temp_state[8]
            self.state[15] = temp_state[10]
            self.state[8] = temp_state[17]
            self.state[10] = temp_state[16]
            self.state[17] = temp_state[7]
            self.state[16] = temp_state[5]
            self.state[7] = temp_state[14]
            self.state[5] = temp_state[15]

        elif turn == 'L':
            self.state[4] = temp_state[6]
            self.state[5] = temp_state[4]
            self.state[7] = temp_state[5]
            self.state[6] = temp_state[7]
            self.state[12] = temp_state[23]
            self.state[14] = temp_state[21]
            self.state[0] = temp_state[12]
            self.state[2] = temp_state[14]
            self.state[16] = temp_state[0]
            self.state[18] = temp_state[2]
            self.state[23] = temp_state[16]
            self.state[21] = temp_state[18]

        elif turn == 'Lp':
            self.state[4] = temp_state[5]
            self.state[5] = temp_state[7]
            self.state[7] = temp_state[6]
            self.state[6] = temp_state[4]
            self.state[12] = temp_state[0]
            self.state[14] = temp_state[2]
            self.state[0] = temp_state[16]
            self.state[2] = temp_state[18]
            self.state[16] = temp_state[23]
            self.state[18] = temp_state[21]
            self.state[23] = temp_state[12]
            self.state[21] = temp_state[14]

        elif turn == 'R':
            self.state[8] = temp_state[10]
            self.state[9] = temp_state[8]
            self.state[11] = temp_state[9]
            self.state[10] = temp_state[11]
            self.state[15] = temp_state[3]
            self.state[13] = temp_state[1]
            self.state[20] = temp_state[15]
            self.state[22] = temp_state[13]
            self.state[19] = temp_state[20]
            self.state[17] = temp_state[22]
            self.state[3] = temp_state[19]
            self.state[1] = temp_state[17]

        elif turn == 'Rp':
            self.state[8] = temp_state[9]
            self.state[9] = temp_state[11]
            self.state[11] = temp_state[10]
            self.state[10] = temp_state[8]
            self.state[15] = temp_state[20]
            self.state[13] = temp_state[22]
            self.state[20] = temp_state[19]
            self.state[22] = temp_state[17]
            self.state[19] = temp_state[3]
            self.state[17] = temp_state[1]
            self.state[3] = temp_state[15]
            self.state[1] = temp_state[13]

        elif turn == 'U':
            self.state[12] = temp_state[14]
            self.state[13] = temp_state[12]
            self.state[15] = temp_state[13]
            self.state[14] = temp_state[15]
            self.state[21] = temp_state[5]
            self.state[20] = temp_state[4]
            self.state[9] = temp_state[21]
            self.state[8] = temp_state[20]
            self.state[1] = temp_state[9]
            self.state[0] = temp_state[8]
            self.state[5] = temp_state[1]
            self.state[4] = temp_state[0]

        elif turn == 'Up':
            self.state[12] = temp_state[13]
            self.state[13] = temp_state[15]
            self.state[15] = temp_state[14]
            self.state[14] = temp_state[12]
            self.state[21] = temp_state[9]
            self.state[20] = temp_state[8]
            self.state[9] = temp_state[1]
            self.state[8] = temp_state[0]
            self.state[1] = temp_state[5]
            self.state[0] = temp_state[4]
            self.state[5] = temp_state[21]
            self.state[4] = temp_state[20]

        elif turn == 'D':
            self.state[16] = temp_state[18]
            self.state[17] = temp_state[16]
            self.state[19] = temp_state[17]
            self.state[18] = temp_state[19]
            self.state[2] = temp_state[6]
            self.state[3] = temp_state[7]
            self.state[10] = temp_state[2]
            self.state[11] = temp_state[3]
            self.state[22] = temp_state[10]
            self.state[23] = temp_state[11]
            self.state[6] = temp_state[22]
            self.state[7] = temp_state[23]

        elif turn == 'Dp':
            self.state[16] = temp_state[17]
            self.state[17] = temp_state[19]
            self.state[19] = temp_state[18]
            self.state[18] = temp_state[16]
            self.state[2] = temp_state[10]
            self.state[3] = temp_state[11]
            self.state[10] = temp_state[22]
            self.state[11] = temp_state[23]
            self.state[22] = temp_state[6]
            self.state[23] = temp_state[7]
            self.state[6] = temp_state[2]
            self.state[7] = temp_state[3]

        elif turn == 'B':
            self.state[20] = temp_state[22]
            self.state[21] = temp_state[20]
            self.state[23] = temp_state[21]
            self.state[22] = temp_state[23]
            self.state[13] = temp_state[11]
            self.state[12] = temp_state[9]
            self.state[4] = temp_state[13]
            self.state[6] = temp_state[12]
            self.state[18] = temp_state[4]
            self.state[19] = temp_state[6]
            self.state[11] = temp_state[18]
            self.state[9] = temp_state[19]

        elif turn == 'Bp':
            self.state[20] = temp_state[21]
            self.state[21] = temp_state[23]
            self.state[23] = temp_state[22]
            self.state[22] = temp_state[20]
            self.state[13] = temp_state[4]
            self.state[12] = temp_state[6]
            self.state[4] = temp_state[18]
            self.state[6] = temp_state[19]
            self.state[18] = temp_state[11]
            self.state[19] = temp_state[9]
            self.state[11] = temp_state[13]
            self.state[9] = temp_state[12]

    def testSequence(self, sequence):

        temp_cube = Cube2x2(self.state.copy())  # .copy() is important!!!

        for turn in sequence:
            temp_cube.move(turn)

        return temp_cube.isSolved()

    def bruteSolve(self):

        if self.isSolved():
            return 'Already solved'

        moves = ['F', 'Fp', 'L', 'Lp', 'R', 'Rp', 'U', 'Up', 'D', 'Dp', 'B', 'Bp']

        for i1 in moves:
            if self.testSequence([i1]): return [i1]
        print('Move length 1 completed')

        for i1 in moves:
            for i2 in moves:
                if self.testSequence([i1,i2]): return [i1,i2]
        print('Move length 2 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    if self.testSequence([i1,i2,i3]): return [i1,i2,i3]
        print('Move length 3 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    for i4 in moves:
                        if self.testSequence([i1,i2,i3,i4]): return [i1,i2,i3,i4]
        print('Move length 4 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    for i4 in moves:
                        for i5 in moves:
                            if self.testSequence([i1,i2,i3,i4,i5]): return [i1,i2,i3,i4,i5]
        print('Move length 5 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    for i4 in moves:
                        for i5 in moves:
                            for i6 in moves:
                                if self.testSequence([i1,i2,i3,i4,i5,i6]): return [i1,i2,i3,i4,i5,i6]
        print('Move length 6 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    for i4 in moves:
                        for i5 in moves:
                            for i6 in moves:
                                for i7 in moves:
                                    if self.testSequence([i1,i2,i3,i4,i5,i6,i7]): return [i1,i2,i3,i4,i5,i6,i7]
        print('Move length 7 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    for i4 in moves:
                        for i5 in moves:
                            for i6 in moves:
                                for i7 in moves:
                                    for i8 in moves:
                                        if self.testSequence([i1,i2,i3,i4,i5,i6,i7,i8]): return [i1,i2,i3,i4,i5,i6,i7,i8]
        print('Move length 8 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    for i4 in moves:
                        for i5 in moves:
                            for i6 in moves:
                                for i7 in moves:
                                    for i8 in moves:
                                        for i9 in moves:
                                            if self.testSequence([i1,i2,i3,i4,i5,i6,i7,i8,i9]): return [i1,i2,i3,i4,i5,i6,i7,i8,i9]
        print('Move length 9 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    for i4 in moves:
                        for i5 in moves:
                            for i6 in moves:
                                for i7 in moves:
                                    for i8 in moves:
                                        for i9 in moves:
                                            for i10 in moves:
                                                if self.testSequence([i1,i2,i3,i4,i5,i6,i7,i8,i9,i10]): return [i1,i2,i3,i4,i5,i6,i7,i8,i9,i10]
        print('Move length 10 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    for i4 in moves:
                        for i5 in moves:
                            for i6 in moves:
                                for i7 in moves:
                                    for i8 in moves:
                                        for i9 in moves:
                                            for i10 in moves:
                                                for i11 in moves:
                                                    if self.testSequence([i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11]): return [i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11]
        print('Move length 11 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    for i4 in moves:
                        for i5 in moves:
                            for i6 in moves:
                                for i7 in moves:
                                    for i8 in moves:
                                        for i9 in moves:
                                            for i10 in moves:
                                                for i11 in moves:
                                                    for i12 in moves:
                                                        if self.testSequence([i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12]): return [i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12]
        print('Move length 12 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    for i4 in moves:
                        for i5 in moves:
                            for i6 in moves:
                                for i7 in moves:
                                    for i8 in moves:
                                        for i9 in moves:
                                            for i10 in moves:
                                                for i11 in moves:
                                                    for i12 in moves:
                                                        for i13 in moves:
                                                            if self.testSequence([i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13]): return [i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13]
        print('Move length 13 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    for i4 in moves:
                        for i5 in moves:
                            for i6 in moves:
                                for i7 in moves:
                                    for i8 in moves:
                                        for i9 in moves:
                                            for i10 in moves:
                                                for i11 in moves:
                                                    for i12 in moves:
                                                        for i13 in moves:
                                                            for i14 in moves:
                                                                if self.testSequence([i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14]): return [i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14]
        return 'Error: no solution found'

    def randomSolve(self, minutes):

        if self.isSolved():
            return 'Already solved'

        actions = ['F', 'Fp', 'L', 'Lp', 'R', 'Rp', 'U', 'Up', 'D', 'Dp', 'B', 'Bp']
        min_len = 1000
        min_sequence = None

        seconds = minutes * 60
        start = time.time()

        while (time.time() - start) < seconds:

            sequence = []
            temp_cube = Cube2x2(self.state.copy())

            for i in range(14):

                action = actions[random.randint(0, 11)]
                sequence.append(action)
                temp_cube.move(action)

                if temp_cube.isSolved():
                    print(sequence)

                    if len(sequence) < min_len:
                        min_len = len(sequence)
                        min_sequence = sequence.copy()

                    break

        if min_sequence == None:
            return 'No solution found'
        else:
            return prune(min_sequence)

    def MCTS_solve(self, model, minutes):

        if self.isSolved():
            return 'Already solved'

        c = 1
        v = 0.1

        tree = [ [Node(self.state, P=model.predict(np.array([encodeOneHot(self.state)]))[0][1:])] ]
        current_pos = [0,0]
        current_node = tree[0][0]

        actions = ['F', 'Fp', 'L', 'Lp', 'R', 'Rp', 'U', 'Up', 'D', 'Dp', 'B', 'Bp']
        seconds = minutes * 60
        start = time.time()

        while (time.time() - start) < seconds:

            print('current_pos: ', current_pos)

            if current_node.children == None:

                print('Adding Children')
                new_children = []

                for action in actions:

                    child_cube = Cube2x2(current_node.state.copy())
                    child_cube.move(action)

                    children_index = 0

                    if current_pos[0] < len(tree) - 1:
                        children_index = len(tree[current_pos[0] + 1])

                    current_node.children = range(children_index, children_index + 12)

                    if child_cube.isSolved():
                        print('---------------------\nSOLVED CUBE ADDED\n---------------------')

                    new_children.append(Node(child_cube.state.copy(), P=model.predict(np.array([encodeOneHot(child_cube.state)]))[0][1:], parent=current_pos[1], previous_action=action))

                if current_pos[0] < len(tree) - 1:
                    tree[current_pos[0] + 1] += new_children
                else:
                    tree.append(new_children)

                max_value = -1000
                for i in current_node.children:
                    value = model.predict(np.array([encodeOneHot(tree[current_pos[0] + 1][i].state)]))[0][0]

                    if value > max_value:
                        max_value = value

                while current_node.parent != None:
                    action_i = actions.index(current_node.previous_action)

                    current_pos = [current_pos[0] - 1, current_node.parent]
                    current_node = tree[current_pos[0]][current_pos[1]]

                    if current_node.W[action_i] < max_value:
                        current_node.W[action_i] = max_value

                    current_node.N[action_i] += 1
                    current_node.L[action_i] -= v

            else:
                print('Selecting Action')

                action_i = 0
                max = -10000

                for i in range(len(actions)):
                    # U(a)
                    summation = 0

                    for j in range(len(actions)):
                        summation += current_node.N[j]

                    U = c * current_node.P[i] * (math.sqrt(summation) / (1 + current_node.N[i]) )
                    Q = current_node.W[i] - current_node.L[i]

                    if U + Q > max:
                        max = U + Q
                        action_i = i

                current_node.L[action_i] += v
                print('Action selected: ', actions[action_i])
                current_pos = [current_pos[0] + 1, current_node.children[action_i]]
                current_node = tree[current_pos[0]][current_pos[1]]

                if Cube2x2(current_node.state).isSolved():

                    while current_node.parent != None:
                        temp_action_i = actions.index(current_node.previous_action)
                        current_pos = [current_pos[0] - 1, current_node.parent]
                        current_node = tree[current_pos[0]][current_pos[1]]
                        current_node.N[temp_action_i] += 1

        for row in range(1, len(tree)):
            for node in range(len(tree[row])):
                if Cube2x2(tree[row][node].state).isSolved():
                    current_pos = [row, node]
                    current_node = tree[row][node]

                    solution = []
                    while current_node.previous_action != None:
                        solution.insert(0, current_node.previous_action)
                        current_pos = [current_pos[0] - 1, current_node.parent]
                        current_node = tree[current_pos[0]][current_pos[1]]

                    return prune(solution)
        return 'No solution found'

    def A_star_solve(self, model):

        if self.isSolved():
            return 'Already solved'

        open = [Node(self.state, h=model.predict(np.array([encodeOneHot(self.state)]))[0][0])]
        closed = []

        actions = ['F', 'Fp', 'L', 'Lp', 'R', 'Rp', 'U', 'Up', 'D', 'Dp', 'B', 'Bp']

        while True:
            print('Running A* Search')
            lowest_cost = 10000
            lowest_cost_node = None

            for node in open:
                cost = 0.6 * node.g + node.h

                if cost < lowest_cost:
                    lowest_cost = cost
                    lowest_cost_node = node

            if Cube2x2(lowest_cost_node.state).isSolved():
                return lowest_cost_node.move_sequence

            open.remove(lowest_cost_node)
            closed.append(lowest_cost_node)

            for a in actions:
                child = Cube2x2(lowest_cost_node.state.copy())
                child.move(a)
                heuristic = None
                if child.isSolved():
                    heuristic = 0
                else:
                    heuristic = model.predict(np.array([encodeOneHot(self.state)]))[0][0]

                child_node = Node(child.state, lowest_cost_node.g + 1, heuristic, lowest_cost_node.move_sequence + [a])

                add_child = True
                for node in closed:
                    if child_node.state == node.state:
                        if child_node.g < node.g:
                            closed.remove(node)
                            open.append(child_node)

                        add_child = False

                if add_child:
                    open.append(child_node)

    def compare(self, model):

        print('\n----------------- bruteSolve -----------------\n')
        start = time.time()
        solution_bruteSolve = self.bruteSolve()
        seconds = time.time() - start
        minutes = seconds / 60

        print('\n----------------- randomSolve -----------------\n')
        solution_randomSolve = self.randomSolve(minutes)

        print('\n----------------- MCTS_solve -----------------\n')
        solution_MCTS_solve = self.MCTS_solve(model, minutes)

        print('\nbruteSolve solution:', solution_bruteSolve)
        print('Solution length:', len(solution_bruteSolve))

        print('\nrandomSolve solution:', solution_randomSolve)
        print('Solution length:', len(solution_randomSolve))

        print('\nMCTS_solve solution:', solution_MCTS_solve)
        print('Solution length:', len(solution_MCTS_solve))

        print('\nExecution time allowed for each method:', seconds, 'seconds')


class Node:

    def __init__(self, state = ['B','B','B','B','O','O','O','O','R','R','R','R','Y','Y','Y','Y','W','W','W','W','G','G','G','G'], g=0, h=0, move_sequence=[]):
        self.state = state
        self.g = g
        self.h = h
        self.move_sequence = move_sequence

def encodeOneHot(state):

    one_hot = []

    for s in state:
        if s == 'B':
            one_hot += [1,0,0,0,0,0]
        elif s == 'O':
            one_hot += [0,1,0,0,0,0]
        elif s == 'R':
            one_hot += [0,0,1,0,0,0]
        elif s == 'Y':
            one_hot += [0,0,0,1,0,0]
        elif s == 'W':
            one_hot += [0,0,0,0,1,0]
        elif s == 'G':
            one_hot += [0,0,0,0,0,1]

    return one_hot

def decodeOneHot(one_hot):

    state = ''

    start = 0
    end = 6

    while end <= len(one_hot):

        section = one_hot[start:end]

        if section == '100000':
            state += 'B'
        elif section == '010000':
            state += 'O'
        elif section == '001000':
            state += 'R'
        elif section == '000100':
            state += 'Y'
        elif section == '000010':
            state += 'W'
        elif section == '000001':
            state += 'G'

        start += 6
        end += 6

    return state

def createModel():

    model = Sequential()
    model.add(Dense(5000, input_dim=144, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    return model

def ADI(minutes, model=None):

    if model == None:

        # Creating the value and policy network
        model = Sequential()
        model.add(Dense(400, input_dim=144, activation='relu'))
        model.add(Dense(200, activation='relu'))
        model.add(Dense(13, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Training Process

    moves = ['F', 'Fp', 'L', 'Lp', 'R', 'Rp', 'U', 'Up', 'D', 'Dp', 'B', 'Bp']
    seconds = minutes * 60
    start = time.time()

    total_samples = 0

    while (time.time() - start) < seconds:

        cube = Cube2x2()
        training_inputs = []

        for i in range(14):
            cube.move(moves[random.randint(0, 11)])
            training_inputs.append(Cube2x2(cube.state.copy()))

        total_samples += 14

        for i in range(len(training_inputs)):
            value_target = -100
            policy_target_move = None

            for move in moves:
                child = Cube2x2(training_inputs[i].state.copy())
                child.move(move)
                value_estimate = model.predict(np.array([encodeOneHot(child.state)]))[0][0]

                if child.isSolved():
                    value_estimate += 1
                else:
                    value_estimate -= 1

                if value_estimate > value_target:
                    value_target = value_estimate
                    policy_target_move = move

            policy_target = None
            if policy_target_move == 'F':
                policy_target = [1,0,0,0,0,0,0,0,0,0,0,0]
            elif policy_target_move == 'Fp':
                policy_target = [0,1,0,0,0,0,0,0,0,0,0,0]
            elif policy_target_move == 'L':
                policy_target = [0,0,1,0,0,0,0,0,0,0,0,0]
            elif policy_target_move == 'Lp':
                policy_target = [0,0,0,1,0,0,0,0,0,0,0,0]
            elif policy_target_move == 'R':
                policy_target = [0,0,0,0,1,0,0,0,0,0,0,0]
            elif policy_target_move == 'Rp':
                policy_target = [0,0,0,0,0,1,0,0,0,0,0,0]
            elif policy_target_move == 'U':
                policy_target = [0,0,0,0,0,0,1,0,0,0,0,0]
            elif policy_target_move == 'Up':
                policy_target = [0,0,0,0,0,0,0,1,0,0,0,0]
            elif policy_target_move == 'D':
                policy_target = [0,0,0,0,0,0,0,0,1,0,0,0]
            elif policy_target_move == 'Dp':
                policy_target = [0,0,0,0,0,0,0,0,0,1,0,0]
            elif policy_target_move == 'B':
                policy_target = [0,0,0,0,0,0,0,0,0,0,1,0]
            elif policy_target_move == 'Bp':
                policy_target = [0,0,0,0,0,0,0,0,0,0,0,1]

            model.fit(np.array([encodeOneHot(training_inputs[i].state)]), np.array([[value_target] + policy_target]), sample_weight=np.array([1/(i+1)]))

    print('Total Samples:', total_samples)

    return model

def DAVI(minutes, model=None):

    if model == None:
        model = createModel()

    actions = ['F', 'Fp', 'L', 'Lp', 'R', 'Rp', 'U', 'Up', 'D', 'Dp', 'B', 'Bp']
    seconds = minutes * 60
    start = time.time()

    iterations = 0
    estimation_model = clone_model(model)

    while (time.time() - start) < seconds:

        iterations += 1
        cube = Cube2x2()
        for i in range(random.randint(1, 14)):
            cube.move(actions[random.randint(0, 11)])

        minimum = 10000
        for a in actions:
            temp_cube = Cube2x2(cube.state.copy())
            temp_cube.move(a)
            cost_to_go = None
            if temp_cube.isSolved():
                cost_to_go = 0
            else:
                cost_to_go = estimation_model.predict(np.array([encodeOneHot(temp_cube.state)]))[0][0]

            temp_value = 1 + cost_to_go
            if temp_value < minimum:
                minimum = temp_value

        loss = (minimum - model.predict(np.array([encodeOneHot(cube.state)]))[0][0])**2
        model.fit(np.array([encodeOneHot(cube.state)]), np.array([[minimum]]))

        if (iterations % 100 == 0) and (loss < 0.05):
            estimation_model = clone_model(model)
            print('estimation_model = clone_model(model)')

    return model

def prune(sequence):

    i = 0
    while i <= len(sequence) - 4:
        if sequence[i] == sequence[i+1] and sequence[i] == sequence[i+2] and sequence[i] == sequence[i+3]:
            for j in range(4):
                sequence.pop(i)
        else:
            i += 1

    i = 0
    while i <= len(sequence) - 3:
        if sequence[i] == sequence[i+1] and sequence[i] == sequence[i+2]:
            if len(sequence[i]) == 1:
                replacement = sequence[i] + 'p'
            else:
                replacement = sequence[i][0]

            for j in range(3):
                sequence.pop(i)

            sequence.insert(i, replacement)
        i += 1

    i = 0
    while i <= len(sequence) - 2:
        if len(sequence[i]) != len(sequence[i+1]) and sequence[i][0] == sequence[i+1][0]:
            for j in range(2):
                sequence.pop(i)
        else:
            i += 1

    return sequence

def test():

    input_state = input('Input the cube state: ')

    start = time.time()

    cube = Cube2x2(list(input_state))
    print(cube.state)

    print('\n--------Deep approximate value iteration--------\n')
    # model = DAVI(1)

    # model.save('model[1min]')
    model = tf.keras.models.load_model('model[1min]')

    print('\n--------Weighted A* Search--------\n')
    solution = cube.A_star_solve(model)

    print('--------------------------\nSolution:', solution, '\n--------------------------')

    print((time.time() - start), 'seconds')

def test_compare():

    input_state = input('Input the cube state: ')

    cube = Cube2x2(list(input_state))
    print(cube.state)

    model = tf.keras.models.load_model('model_620min')

    cube.compare(model)

def train():

    start = time.time()

    print('\n--------AUTODIDACTIC ITERATION--------\n')
    model = ADI(620)
    model.save('model_620min')

    print((time.time() - start), 'seconds')

test()
