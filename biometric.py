from reedMuller import FirstOrderReedMuller
from tensorProductCode import TensorProductCode
from utils import generateRandomBitVec, generateBitErrors
import hashlib
import numpy as np


class BiometricSystem:
    def __init__(self, code):
        self.code = code
        self.hashFunction = hashlib.sha3_512()

    def enrollment(self, irisCode):
        key = generateRandomBitVec(self.code.k)
        encodedKey = self.code.encode(key)
        print("KEY CODEWORD: ", encodedKey)
        self.hashFunction.update(encodedKey)

        self.keyHash = self.hashFunction.hexdigest()
        print("KEY CODEWORD HASH: ", self.keyHash)
        self.lock = np.logical_xor(encodedKey, irisCode)

    def authentication(self, irisCode):
        unlockedKey = np.logical_xor(self.lock, irisCode)
        decodedList = self.code.listDecode(unlockedKey, eps=0.001)

        for codeword in decodedList:
            print(codeword)
            self.hashFunction.update(codeword)
            codewordHash = self.hashFunction.hexdigest()
            print(codewordHash)
            if self.keyHash == codewordHash:
                return True
        
        return False


if __name__ == "__main__":
    system = BiometricSystem(TensorProductCode(FirstOrderReedMuller(6), FirstOrderReedMuller(5)))

    trueIrisCode = generateRandomBitVec(2048)
    print("TRUE IRIS CODE:", trueIrisCode)
    system.enrollment(trueIrisCode)

    trueIrisCodeWithErrors = generateBitErrors(trueIrisCode, 50)
    falseIrisCode = generateRandomBitVec(2048)

    print("TRUE IRIS CODE AUTH RESULT:", system.authentication(trueIrisCode))
    # print("TRUE IRIS WITH ERRORS AUTH RESULT:", system.authentication(trueIrisCodeWithErrors))
    # print("FALSE IRIS CODE AUTH RESULT:", system.authentication(falseIrisCode))
