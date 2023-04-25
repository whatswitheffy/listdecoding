from reedMuller import FirstOrderReedMuller
from tensorProductCode import TensorProductCode
from utils import seqGenerator, generateBitErrors
import hashlib
import numpy as np


class BiometricSystem:
    def __init__(self, lenOfIris, lenOfKey):
        self.irisCode, self.key = seqGenerator(lenOfIris, lenOfKey)
        self.codeProduct = TensorProductCode(FirstOrderReedMuller(2), FirstOrderReedMuller(2))
        self.encodedKey = self.codeProduct.encode(self.key)
        self.hash = self.keyHash()
        self.lock = self.irisAndCodewordXOR()

    def keyHash(self):
        hash = hashlib.sha3_512()
        hash.update(self.key)
        print("Hash of key:", hash.hexdigest())
        return hash.hexdigest()
    
    def irisAndCodewordXOR(self):
        lock = np.logical_xor(self.irisCode, self.encodedKey)
        print("LOCK: ", lock.astype(int) )
        return lock
            

if __name__ == "__main__":
    biom = BiometricSystem(16, 9)
    # decoded_list = rm.listDecode(biom.codewordWithErrors, 0.1)
    # for decoded_codeword in decoded_list:
    #     print(decoded_codeword)
    #     if np.allclose(decoded_codeword, biom.codeword):
    #         print("EQUAL")