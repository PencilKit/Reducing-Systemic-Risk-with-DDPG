import numpy as np
import numpy.matlib
from scipy.optimize import linprog

from itertools import product


class Network_Object(object):

    def __init__(self, n_agents, m_layers, total_loans_left, total_borrowing_left):
        
        self.n_agents = n_agents
        self.m_layers = m_layers

        self.m_total_loans = total_loans_left
        self.m_total_borrowing = total_borrowing_left
        self.B = []

    def solve_linprog(self, alpha):
        row_total = self.m_total_loans[alpha]
        col_total = self.m_total_borrowing[alpha]

        m = np.size(row_total)
        n = np.size(col_total)

        col1 = np.matlib.repmat(np.arange(m), 1, n)[0] + 1
        col1 = np.concatenate((col1, np.repeat(np.arange(n)+m, m) + 1))
        col2 = np.matlib.repmat(np.arange(m*n), 1, 2)[0] + 1

        
        mat_pairs = np.array([col1, col2]).T

        Aeq = np.zeros((n+m+1, n*m))
        for i in range(n+m):
            for j in range(m*n):
                for k in range(m*n*2):
                    if np.all(mat_pairs[k] == np.array([i+1, j+1])):
                        Aeq[i, j] += 1
    
        Aeq[-1, :] = np.identity(n).reshape(n*m)

        # np.fill_diagonal(Aeq, 0)

        beq = np.concatenate((row_total, col_total))
        beq = np.concatenate((beq, np.array([0])))
        lb = 0
        ub = np.max(beq)
        
        B = np.zeros((m*n, m*n))
        for k in range(m*n):
            f = np.zeros(m*n)
            f[k] = -1
            # B[:, k] = linprog(f, A_eq=Aeq.T, b_eq=beq, method='simplex', options={'rr': False})['x']
            B[:, k] = linprog(f, A_eq=Aeq, b_eq=beq, method='simplex')['x']

        self.B.append(B)

    def make_experience(self):

        n = self.n_agents

        exp_mat = []
        for alpha in range(self.m_layers):
            x = np.random.dirichlet(np.ones(n*n))
            exp_mat.append(np.matmul(self.B[alpha], x).reshape((n, n)))

        return exp_mat



N_AGENTS = 3
M_LAYERS = 1
TOTAL_LOANS_LEFT = np.array([[2, 2, 2]])
TOTAL_BORROWING_LEFT = np.array([[1, 3, 2]])
# TOTAL_LOANS_LEFT = np.array([[500, 200]])
# TOTAL_BORROWING_LEFT = np.array([[300, 400]])
no = Network_Object(
    n_agents=N_AGENTS,
    m_layers=M_LAYERS,
    total_loans_left=TOTAL_LOANS_LEFT,
    total_borrowing_left=TOTAL_BORROWING_LEFT
)

no.solve_linprog(0)

no.make_experience()

print('done')