import itertools
import numpy as np
from numpy.typing import NDArray
from reedMuller import FirstOrderReedMuller
from utils import hammingDistNormalized, pickRandomSubset, generateBitVec, generateBitErrors


class TensorProductCode:
    def __init__(self, code1: FirstOrderReedMuller, code2: FirstOrderReedMuller):
        self.code1 = code1
        self.code2 = code2
        self.n = self.code1.n * self.code2.n
        self.k = self.code1.k * self.code2.k
        self.d = self.code1.d * self.code2.d
        self.eta = self.code1.delta * self.code2.delta
        self.generatorMatrix = np.kron(self.code1.generatorMatrix, self.code2.generatorMatrix)

    def encode(self, message):
        res = (self.generatorMatrix.T * message).T
        codeword = np.zeros(res.shape[1])
        for i in range(res.shape[0]):
            codeword = np.logical_xor(codeword, res[i]).astype(int)       
        return codeword

    def listDecode(self, receivedWord: NDArray, eps: float) -> list[NDArray]:
        m1 = 3
        m2 = 3

        R = np.reshape(receivedWord, (self.code2.n, self.code1.n))

        S = pickRandomSubset(list(range(self.code2.n)), m1)
        T = pickRandomSubset(list(range(self.code1.n)), m2)

        L = []
        allA = self._generateAllAssignments(S, T)
        for A in allA:
            B, S_success = self._phase1(R, eps, A, S, T)
            D, T_success = self._phase2(R, eps, S, B, S_success)
            E, U_success = self._phase3(R, eps, D, T_success)
            C = self._phase4(E, U_success).flatten()
            # print(hammingDistNormalized(C.flatten(), R.flatten()), self.eta - 3 * eps)
            if hammingDistNormalized(C, receivedWord) <= self.eta - 3 * eps:
                L.append(C)

        return L

    def _generateAllAssignments(self, S, T):
        for flat in itertools.product([1, 0], repeat=len(S)*len(T)):
            submatrix = np.reshape(flat, (len(S), len(T)))
            A = np.zeros((self.code2.n, self.code1.n))
            A[np.ix_(S, T)] = submatrix
            yield A

    def _phase1(self, R, eps, A, S, T):
        B = np.zeros((max(S) + 1, self.code1.n))
        S_success = []
        for s in S:
            L_s = self.code1.listDecode(R[s, :], eps)
            found = next((c for c in L_s if (c[T] == A[s, T]).all()), None)
            if found is not None:
                B[s, :] = found
                S_success.append(s)
        return B, S_success
    
    def _phase2(self, R, eps, S, B, S_success):
        D = np.zeros((self.code2.n, self.code1.n))
        T_success = []
        for t in range(self.code1.n):
            L_t = self.code2.listDecode(R[:, t])
            found = next(
                (c for c in L_t 
                 if hammingDistNormalized(c[S_success], B[S_success, t]) < eps * len(S)
                ), 
                None
            )
            if found is not None:
                D[:, t] = found
                T_success.append(t)
        return D, T_success

    def _phase3(self, R, eps, D, T_success):
        E = np.zeros((self.code2.n, self.code1.n))
        U_success = []
        for s in range(self.code2.n):
            L_s = self.code1.listDecode(R[s, :])
            found = next(
                (c for c in L_s 
                 if hammingDistNormalized(c[T_success], D[s, T_success]) < eps * self.code1.n
                ), 
                None
            )
            if found is not None:
                E[s, :] = found
                U_success.append(s)
        return E, U_success
    
    def _phase4(self, E, U_success):
        C = np.zeros((self.code2.n, self.code1.n))
        for t in range(self.code1.n):
            C[:, t] = self.code2.decodeErasuresToCodeword(E[:, t], U_success)
        return C


if __name__ == "__main__":
    rm1 = FirstOrderReedMuller(3)
    rm2 = FirstOrderReedMuller(3)

    product = TensorProductCode(rm1, rm2)

    message = generateBitVec(product.k)
    print("MESSAGE: ", message)

    word = product.encode(message)
    print("ENCODED: ", word, " | LEN: ", len(word))

    w_err = generateBitErrors(word, 0)
    print("ERRORED: ", w_err)

    decoded_list = product.listDecode(w_err, 0.01)
    print("DECODED: ") 
    for decoded in decoded_list:
        print(decoded)
        if (decoded == word).all():
            print("^EQUAL^")
    
