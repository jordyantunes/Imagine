import sys; sys.path.append('../')
from gym import Env

class EnvController():
    __instance = None
    env = None

    @staticmethod 
    def getInstance():
        """ Static access method. """
        if EnvController.__instance == None:
           raise Exception("Class not initialized!")
        return EnvController.__instance
    def __init__(self, env: Env):
       """ Virtually private constructor. """
       if EnvController.__instance != None:
          raise Exception("This class is a singleton!")
       else:
          EnvController.__instance = self
          self.env = env