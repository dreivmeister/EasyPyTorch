import unittest
from src.inference import Inference



class TrainingClassTest(unittest.TestCase):
    def create_class(self):
        I = Inference()
        assert I is not None


if __name__=='__main__':
    unittest.main()