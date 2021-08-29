import numpy as np


def solver_k_eq_n(S : np.ndarray):
	eigv = np.linalg.eigvals(S)
	mu = 1 / eigv

	val = eigv @ mu - np.sum(np.log(mu))

	return val


def solver_k_eq_0(S : np.ndarray):
	K = np.diag(1 / S.diagonal())
	return K


def solve_cf(S : np.ndarray, k: int):

	b = np.zeros(k + 1)
	b[0] = 1

	n = len(S)
	U = np.zeros((n, n))

	for i in range(n):
		if False and i % ((n // 100) + 1) == 0:
			print('%s%%' % (i // ((n // 100) + 1)))

		relevant_S = S[i: i + k + 1, i: i + k + 1]

		x = np.linalg.solve(relevant_S, b[:len(relevant_S)])
		alpha = 1 / np.sqrt(x[0])

		U[i, i:i+k+1] = alpha * x

	K = U.T @ U
	return K


def solve(S : np.ndarray, k: int):
	n = len(S)
	if k == 0:
		return solver_k_eq_0(S)
	if k == n - 1:
		return solver_k_eq_n(S)
	else:
		return solve_cf(S, k)
