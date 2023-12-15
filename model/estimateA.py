import numpy as np
from collections import Counter

class EstimateAdj():
    def __init__(self, data, arg):
        self.num_class = data.num_class
        self.num_node = data.num_node
        self.idx_train = data.idx_train
        self.label = data.y.cpu().numpy()
        self.adj = data.adj.to_dense().cpu().numpy()

        self.output = None
        self.iterations = 0
        self.E = 0
        self.N = 0
        # self.homophily = data.homophily
        self.dataset = arg.dataset
        self.labelrate = arg.labelrate

    def reset_obs(self):
        self.N = 0
        self.E = np.zeros((self.num_node, self.num_node), dtype=np.int64)

    def update_obs(self, output):
        self.E += output
        self.N += 1

    def revise_pred(self):
        for j in range(len(self.idx_train)):
            self.output[self.idx_train[j]] = self.label[self.idx_train[j]]

    def E_step(self, Q):
        an = Q * self.E
        an = np.triu(an, 1).sum()
        bn = (1 - Q) * self.E
        bn = np.triu(bn, 1).sum()
        ad = Q * self.N
        ad = np.triu(ad, 1).sum()
        bd = (1 - Q) * self.N
        bd = np.triu(bd, 1).sum()

        alpha = an * 1. / (ad)
        beta = bn * 1. / (bd)

        O = np.zeros((self.num_class, self.num_class))

        n = []
        counter = Counter(self.output)
        for i in range(self.num_class):
            n.append(counter[i])

        a = self.output.repeat(self.num_node).reshape(self.num_node, -1)
        for j in range(self.num_class):
            c = (a == j)
            for i in range(j + 1):
                b = (a == i)
                O[i, j] = np.triu((b & c.T) * Q, 1).sum()
                if self.dataset == 'uai' and self.labelrate == 20:
                    if i == j:
                        O[j, j] = (2. + 0.00000001) / ((n[j] * (n[j] - 1)) * O[j, j] + 0.00000001)
                    else:
                        O[i, j] = (1. + 0.00000001) / ((n[i] * n[j]) * O[i, j] + 0.00000001)
                else:
                    if i == j:
                        O[j, j] = 2. / (n[j] * (n[j] - 1)) * O[j, j]
                    else:
                        O[i, j] = 1. / (n[i] * n[j]) * O[i, j]
        return (alpha, beta, O)

    def M_step(self, alpha, beta, O):

        O += O.T - np.diag(O.diagonal())
        row = self.output.repeat(self.num_node)
        col = np.tile(self.output, self.num_node)
        tmp = O[row, col].reshape(self.num_node, -1)

        p1 = tmp * np.power(alpha, self.E) * np.power(1 - alpha,
                                                      self.N - self.E)
        p2 = (1 - tmp) * np.power(beta, self.E) * np.power(1 - beta, self.N - self.E)
        Q = p1 * 1. / (p1 + p2 * 1.)
        return Q

    def EM(self, output, tolerance=.000001):
        alpha_p = 0
        beta_p = 0

        self.output = output
        self.revise_pred()

        beta, alpha = np.sort(np.random.rand(2))
        O = np.triu(np.random.rand(self.num_class, self.num_class))

        Q = self.M_step(alpha, beta, O)

        while abs(alpha_p - alpha) > tolerance or abs(beta_p - beta) > tolerance:
            alpha_p = alpha
            beta_p = beta
            alpha, beta, O = self.E_step(Q)
            Q = self.M_step(alpha, beta, O)
            self.iterations += 1

        # if self.homophily > 0.5:
            # Q += self.adj
        return (alpha, beta, O, Q, self.iterations)