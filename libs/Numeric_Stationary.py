import networkx as nx
import numpy as np
import scipy.sparse.linalg as spsl
import scipy.sparse as sps
import timeit
import random
import itertools
# import libs.Methods as Methods


BLUE = 1
RED = 0

def binary_list_to_decimal(binary_list):
    """
    Convert a list of 0s and 1s representing a binary number to a base 10 integer.
    
    :param binary_list: List[int] - A list of 0s and 1s.
    :return: int - The decimal representation of the binary number.
    """
    
    decimal_number = 0
    for bit in binary_list:
        decimal_number = (decimal_number << 1) | bit
    return decimal_number


class Numeric_Solver_Stationary:
    def __init__(self, nxGraph : nx.Graph, coefficient : float = 1, mutation_rate : float = 0.01, replacer = True):
        self.nxGraph = nxGraph
        self.coefficient = coefficient
        self.mutation_rate = mutation_rate
        self.powers = []
        self.matrix = None
        self.replacer = replacer
        self.num_mutants_in_state = []
    
        for k in range(self.__number_of_nodes()):
            self.powers.append(2**(self.__number_of_nodes()-1-k))

        self.__init_matrix()
    
    def __number_of_nodes(self):
        return self.nxGraph.number_of_nodes()
    def __get_power(self, exponent):
        return self.powers[exponent]
    
    def __init_matrix(self):
        number_of_states = 2**self.__number_of_nodes()
        self.matrix = sps.lil_matrix((number_of_states,number_of_states))
        for state_vector in itertools.product((0,1),repeat=self.__number_of_nodes()):
            fitness_ind=[]
            total_fitness = 0
            num_of_mutants = 0 
            state_number = binary_list_to_decimal(state_vector)
            for k in range(self.__number_of_nodes()):
                if state_vector[k]==BLUE:
                    fitness_ind.append(self.coefficient)
                    total_fitness += self.coefficient
                    num_of_mutants += 1
                else:
                    fitness_ind.append(1)
                    total_fitness += 1
            self.num_mutants_in_state.append(num_of_mutants)
            for node in self.nxGraph.nodes():
                secondary_nodes_blue = []
                secondary_nodes_red = []

                neighbors = list(self.nxGraph.neighbors(node))
                # print(neighbors)
                for neighbor in neighbors:
                    if state_vector[neighbor] == RED:
                        secondary_nodes_red.append(neighbor)

                        
                    if state_vector[neighbor] == BLUE:
                        secondary_nodes_blue.append(neighbor)

                
                for secondary_node in neighbors:
                    if state_vector[node] == BLUE:
                        if self.replacer: 
                            if state_vector[secondary_node] == RED:
                                new_state = state_number + self.__get_power(secondary_node)
                                change = (1 -self.mutation_rate)*fitness_ind[node]/total_fitness/len(secondary_nodes_red)
                                
                                self.matrix[state_number, new_state] +=change
                                #I AM NOT SURE ABOUT LINE BELOW
                                self.matrix[state_number, state_number] -= change
                            if state_vector[secondary_node] == BLUE:
                                new_state = state_number - self.__get_power(secondary_node)
                                if len(secondary_nodes_red) > 0:
                                    # do not overwrite existing entries; skip adding mutation to same-color neighbor
                                    pass
                                else: 
                                    change =  self.mutation_rate*fitness_ind[node]/total_fitness/len(secondary_nodes_blue)
                                    self.matrix[state_number, new_state] += change
                                    #I AM NOT SURE ABOUT LINE BELOW
                                    self.matrix[state_number, state_number] -= change
                        else:
                            if state_vector[secondary_node] == RED:
                                new_state = state_number + self.__get_power(secondary_node)
                                change = (1 -self.mutation_rate)*fitness_ind[node]/total_fitness/len(neighbors)
                                self.matrix[state_number, new_state] += change
                                self.matrix[state_number, state_number] -= change

                            else:
                                new_state = state_number - self.__get_power(secondary_node)
                                change = self.mutation_rate*fitness_ind[node]/total_fitness/len(neighbors)
                                self.matrix[state_number, new_state] += change
                                self.matrix[state_number, state_number] -= change

                    if state_vector[node] == RED:
                        if state_vector[secondary_node] == BLUE:
                            new_state = state_number - self.__get_power(secondary_node)
                            change = (1 -self.mutation_rate)*fitness_ind[node]/total_fitness/len(neighbors)
                            self.matrix[state_number, new_state] += change
                            self.matrix[state_number, state_number] -= change
                        if state_vector[secondary_node] == RED:
                            new_state = state_number + self.__get_power(secondary_node)
                            change = self.mutation_rate*fitness_ind[node]/total_fitness/len(neighbors)
                            self.matrix[state_number, new_state] += change
                            self.matrix[state_number, state_number] -= change

                # for secondary_node in neighbors:
                #     if state_vector[node] != state_vector[secondary_node]:
                #         if state_vector[node] == RED:
                #             new_state = state_number - self.__get_power(secondary_node)
                #         else:
                #             new_state = state_number + self.__get_power(secondary_node)
                #         self.matrix[state_number, new_state] += mut[node]/float(total_fitness*len(secondary_nodes))
                #         self.matrix[state_number, state_number] -= mut[node]/float(total_fitness*len(secondary_nodes))
      
        for i in range(number_of_states):
            self.matrix[i,i] += 1
        # print(self.matrix[0,1])
        # print(self.mutation_rate/5)
        # print(self.matrix[number_of_states-1,number_of_states-2])

        # quick stochasticity check
        M = self.matrix.tocsr()
        row_sums = np.asarray(M.sum(axis=1)).flatten()
        min_entry = float(M.data.min()) if M.nnz > 0 else 0.0
        neg_count = int((M.data < -1e-12).sum()) if M.nnz > 0 else 0
        tol_rtol = 1e-8
        tol_atol = 1e-10
        rows_close = np.isclose(row_sums, 1.0, rtol=tol_rtol, atol=tol_atol)

        self.is_stochastic = bool(rows_close.all() and neg_count == 0)
        self.row_sums = row_sums
        self.min_entry = min_entry
        self.negative_entries_count = neg_count

        if self.is_stochastic:
            # print("Matrix appears stochastic: all rows sum to 1 within tolerance and no negative entries.")
            pass
        else:
            bad_rows = np.where(~rows_close)[0]
            print(f"Matrix NOT stochastic: min_entry={self.min_entry:.3e}, negative_entries={self.negative_entries_count}, "
                  f"rows_not_summing_to_1={len(bad_rows)} (showing up to 10 indices: {bad_rows[:10]})")
    
    def find_stationary_distribution(self):
        """
        Compute stationary distribution π for transition matrix self.matrix (sparse).
        Solves πP = π with sum(π)=1 by solving (P^T - I)^T π^T = 0 with one row replaced.
        """
        n = self.matrix.shape[0]
        # First attempt: sparse linear solve for (P^T - I)^T pi^T = 0 with sum(pi)=1

        PT = self.matrix.transpose().tocsc()
        A = PT - sps.eye(n, format='csc')
        A = A.tolil()
        # replace last equation with sum(pi) = 1 (use explicit index instead of negative index)
        A[n-1, :] = 1
        A = A.tocsc()
        b = np.zeros(n, dtype=float)
        b[n-1] = 1.0
        pi_lin = spsl.spsolve(A, b)
        pi_lin = np.asarray(pi_lin, dtype=float)


        return pi_lin

    def find_expected_number_of_mutants(self):
        stationary_distribution = self.find_stationary_distribution()
        expected_num_of_mutants = 0
        # print(self.num_mutants_in_state)
        # print(stationary_distribution)
        for i in range(len(stationary_distribution)):
            expected_num_of_mutants += stationary_distribution[i]*self.num_mutants_in_state[i]
        return expected_num_of_mutants


    def solve(self ):
        size=self.matrix.shape[1]
        right_hand_side=np.zeros((2,size))
        right_hand_side[0][size-1]=1
        right_hand_side[1] = np.full(size, -1)
        right_hand_side[1][size-1] = 0
        right_hand_side[1][0] = 0
        right_hand_side = np.transpose(right_hand_side)
        self.__solution = spsl.spsolve(sps.csr_matrix(self.matrix),right_hand_side)
        
        self.__fixation_probabilities = []
        self.__absorption_times = []
        
        ind=1
        sum_fixation_probabilities = 0
        sum_absorption_times = 0
        while ind<size:
            self.__fixation_probabilities.append(float(self.__solution[ind][0]))
            self.__absorption_times.append(float(self.__solution[ind][1]))
            sum_fixation_probabilities += float(self.__solution[ind][0])
            sum_absorption_times += float(self.__solution[ind][1])
            ind=2*ind
        self.__fixation_probabilities = self.__fixation_probabilities[::-1]
        self.__absorption_times = self.__absorption_times[::-1]    
        self.__average_fixation_probability = sum_fixation_probabilities/len(self.__fixation_probabilities)
        self.__average_absorption_time = sum_absorption_times/len(self.__absorption_times)
    def __color_list_to_decimal(self, color_list):
        number = 0
        for i in range(len(color_list)):
            if color_list[i] == 'blue' or color_list[i] == BLUE:
                number += self.__get_power(i)
        return number
    
    def get_average_fixation_probability(self):
        return self.__average_fixation_probability
    
    def get_average_absorption_time(self):
        return self.__average_absorption_time
    
    def get_fixation_probabilities(self, list_of_blues :list = None , color_list : list = None):
        
        if list_of_blues is not None:
            index = 0
            for node in list_of_blues:
                index += self.__get_power(node)
            return float(self.__solution[index][0])
        elif color_list is not None:
            return float(self.__solution[self.__color_list_to_decimal(color_list)][0])
        else:
            return self.__fixation_probabilities
    
    def get_absorption_times(self, list_of_blues :list = None , color_list : list = None):
        if list_of_blues is not None:
            index = 0
            for node in list_of_blues:
                index += self.__get_power(node)
            return float(self.__solution[index][1])
        elif color_list is not None:
            return float(self.__solution[self.__color_list_to_decimal(color_list)][1])
        else:
            return self.__absorption_times

if __name__ == "__main__":
    import numpy as np
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import spsolve
    from scipy.sparse import lil_matrix
    import networkx as nx

    G = nx.cycle_graph(10)
    numeric_solver = Numeric_Solver_Stationary(G, coefficient=1, mutation_rate=0.5)
    # quick diagnostics to help debug stationary distribution issues
    M = numeric_solver.matrix.tocsr()
    row_sums = np.array(M.sum(axis=1)).flatten()
    print("row sums: min=", row_sums.min(), "max=", row_sums.max(), "mean=", row_sums.mean())
    # check for negative entries
    min_entry = M.data.min() if M.nnz > 0 else 0.0
    neg_count = (M.data < -1e-15).sum() if M.nnz > 0 else 0
    print("matrix entries: min=", float(min_entry), "nnz=", M.nnz, "negative_entries=", int(neg_count))

    print(numeric_solver.find_expected_number_of_mutants())
                    
