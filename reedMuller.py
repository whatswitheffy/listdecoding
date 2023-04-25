import itertools
import numpy as np
# import sympy
import scipy.linalg
from numpy.typing import NDArray
from utils import minHammingDist, hammingDist, generateBitErrors


class FirstOrderReedMuller:
    def __init__(self, m):
        self.m = m
        self.n = 2 ** m
        self.k = m + 1
        self.d = 2 ** (m - 1)
        self.delta = self.d / self.n
        self.generatorMatrix = self.matrixGen()
        # self.parityCheckMatrix = np.array(sympy.Matrix(self.generatorMatrix).nullspace()) % 2
        self.hadamardMatrix = scipy.linalg.hadamard(self.n)

    def vectorGen(self):
        return [tuple(reversed(x)) for x in itertools.product([0, 1], repeat = self.m)]

    def matrixGen(self):
        vectors = self.vectorGen()
        matrix = np.array(list(reversed(vectors))).T
        return np.vstack([[1] * len(vectors), matrix])
    
    def getCodewordsList(self):
        codewordsList = []
        for msg in [np.array(x, dtype=int) for x in itertools.product([0, 1], repeat = self.k)]:
            codewordsList.append(self.encode(msg))
        return codewordsList
    
    def encode(self, message: NDArray) -> NDArray:
        res = (self.generatorMatrix.T * message).T
        # print("RESULT OF APPLIC:\n", res)
        codeword = np.zeros(res.shape[1])
        for i in range(res.shape[0]):
            codeword = np.logical_xor(codeword, res[i]).astype(int)       
        # print("Applicated code: ", codeword)
        return codeword
    
    def decodeToCodeword(self, receivedWord: NDArray) -> NDArray:
        Y = 2 * receivedWord - 1
        YH = Y @ self.hadamardMatrix
        idx = np.argmax(np.abs(YH))
        C = self.hadamardMatrix[idx] if YH[idx] > 0 else -self.hadamardMatrix[idx]
        return (C + 1) // 2
    
    def decodeErasuresToCodeword(self, receivedWord: NDArray, uncorruptedPositions: list[int]) -> NDArray:
        erasuresPositions = [i for i in range(len(receivedWord)) if i not in uncorruptedPositions]
        y1 = np.copy(receivedWord)
        y2 = np.copy(receivedWord)
        y1[erasuresPositions] = 0
        y2[erasuresPositions] = 1
        c1 = self.decodeToCodeword(y1)
        c2 = self.decodeToCodeword(y2)
        dist1 = hammingDist(c1[uncorruptedPositions], receivedWord[uncorruptedPositions])
        dist2 = hammingDist(c2[uncorruptedPositions], receivedWord[uncorruptedPositions])
        return c1 if dist1 < dist2 else c2

    def listDecode(self, receivedWord: NDArray, eps=0.5) -> list[NDArray]:
        T = (1 - eps) * self.d

        L = [[0] * (2 ** self.m), [int(i / (2 ** (self.m - 1)) % 2) for i in range(2 ** self.m)]]
        for j in range(1, self.m):
            newL = []     
            xj = [int(i / (2 ** (self.m - j - 1)) % 2) for i in range(2 ** self.m)]
            for prefix in L:
                D = 0
                for i in range(2 ** (self.m - j)):
                    D += minHammingDist(prefix[i:2 ** self.m:2 ** (self.m - j)], receivedWord[i:2 ** self.m:2 ** (self.m - j)])

                if D <= T:
                    newL.append(prefix)
                    newL.append([(x + y) % 2 for (x, y) in zip(prefix, xj)])
            L = newL

        resL = []
        for codeword in L:
            dist = hammingDist(codeword, receivedWord)
            if dist > T:
                resL.append(np.array([(bit + 1) % 2 for bit in codeword]))
            else:
                resL.append(np.array(codeword))

        return resL
    
        
# if __name__ == "__main__":
#     import random


#     rm = FirstOrderReedMuller(4)
#     message = np.array([1, 0, 1, 0, 1])
#     print(f"message: {message}")

#     codeword = rm.encode(message)
#     print(f"codeword: {codeword}")

#     errors_number = 4
#     erasures_number = 7
#     true_cnt = 0
#     all_count = 1000
#     for _ in range(all_count):
#         # codeword_with_errors = generateBitErrors(codeword, err_number)
#         # print(f"codeword with errors: {codeword_with_errors}")
#         uncorruptedPositions = random.sample(range(len(codeword)), len(codeword) - erasures_number)
#         decoded = rm.decodeErasuresToCodeword(codeword, uncorruptedPositions)
#         print(f"decoded codeword: {decoded}")
#         if (decoded == codeword).all():
#             true_cnt += 1

#     print(f"TOTAL: {true_cnt}/{all_count}")