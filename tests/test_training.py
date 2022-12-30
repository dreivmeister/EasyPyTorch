import unittest
from src.training import Training

class TrainingClassTest(unittest.TestCase):
    def create_class(self):
        T = Training()
        assert T is not None


if __name__=='__main__':
    unittest.main()
    
    
#how to run from:
#C:\Users\DELL User\Desktop\EasyPyTorch\EasyPyTorch>:
#py -m unittest tests.test_training.TrainingClassTest.create_class