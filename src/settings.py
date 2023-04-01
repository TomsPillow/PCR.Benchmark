import yaml

class ModelSettings():
    def __init__(self,path="./configs/modelSettings.yml"):
        f=open(path)
        self.modelSettings=yaml.load(f)
    
    def toDict(self):
        return self.modelSettings

class DatasetSettings():
    def __init__(self,path='./configs/datasetSettings.yml'):
        f=open(path)
        self.datasetSettings=yaml.load(f)
    
    def toDict(self):
        return self.modelSettings

class CheckpointSettings():
    def __init__(self,path='./configs/checkpointSettings.yml'):
        f=open(path)
        self.checkpointSettings=yaml.load(f)
    
    def toDict(self):
        return self.checkpointSettings
    
