import numpy as np
import datetime

def timeToID() -> str:
    return str(datetime.date().year())+str(datetime.date().month)+str(datetime.date().day())\
            +str(datetime.time().hour())+str(datetime.time().minute())+str(datetime.time().second())

class Transform():
    def __init__(self,R,t):
        self.R=R
        self.t=t
    
    def toDict(self):
        return {'R':self.R,'t':self.t}


class PCRTasksController():
    def __init__(self, total):
        self.completed=[]
        self.working=[]
        self.total=total
        
    def submitTask(self,task):
        self.working.append(task)
        
    def submitEstimation(self, estTransform):
        if(len(self.working)>0):
            self.working[0].setEstTransform(estTransform)
            self.completed.append(self.working[0])
            self.working.pop(0)

class PCRTask():
    def __init__(self,srcPcPath,tgtPcPath,gtTransform):
        self.taskID=timeToID()
        self.srcPcPath=srcPcPath
        self.tgtPcPath=tgtPcPath
        self.gtTransform=gtTransform
        self.estTransform=np.zeros_like(gtTransform)

    def setEstTransform(self,estTransform):
        self.estTransform=estTransform
    
    def toDict(self):
        return {
            'taskID':self.taskID,
            'srcPcPath':self.srcPcPath,
            'tgtPcPath':self.tgtPcPath,
            'gtTransform':self.gtTransform.toDict(),
            'estTransform':self.estTransform.toDict(),
            }
    
