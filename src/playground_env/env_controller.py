import sys; sys.path.append('../')
from gym import Env
from deepdiff import DeepDiff

class EnvController():
   __instance = None
   env = None
   stage:str = 'creation'

   @staticmethod 
   def getInstance():
      """ Static access method. """
      if EnvController.__instance == None:
         raise Exception("Class not initialized!")
      return EnvController.__instance
   def __init__(self, env: Env):
      """ Virtually private constructor. """
      if EnvController.__instance != None:
      #  diff = DeepDiff(self.env, env, ignore_order=True) 
      #  print(diff)
         if self.env is not None:
            raise Exception("This class is a singleton!")
      else:
         EnvController.__instance = self
         self.env = env