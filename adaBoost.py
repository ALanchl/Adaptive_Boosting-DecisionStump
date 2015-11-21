__author__ = 'Chandu'
import sys,math,operator,os

class adaboost:
    trainingData=[]
    boostItr = None
    testingData = []
    trEntropy = None

    def __init__(self,trFile,testFile,boostItr):
        self.boostItr = boostItr
        self.trainingData = self.readTrainingFile(trFile)
        self.testingData = self.readTestingFile(testFile)
        self.trEntropy = self.calculateEntropy(self.trainingData)

    def readTrainingFile(self,name):
        lines=None
        with open(name,"r") as reader:
            lines = reader.readlines()
        lines =lines[0].strip().replace("\n","").split("\r")

        for i in range(0,len(lines)):
            line = lines[i]
            line = line.strip()+"\t"+str(1.0/len(lines))
            lines[i] = line

        return lines

    def readTestingFile(self,name):
        lines=None
        with open(name,"r") as reader:
            lines = reader.readlines()
        return lines[0].strip().replace("\n","").split("\r")

    def calculateEntropy(self,data):
        size = len(data)
        classWeight = {}
        totalWeight = 0.0
        for line in data:
            data=line.strip().split("\t")
            classWeight.setdefault(data[0],0.0)
            classWeight[data[0]]+=float(data[len(data)-1])
            totalWeight +=float(data[len(data)-1])

        entropy = 0.0
        for k,v in classWeight.items():
            entropy += ((v/totalWeight)*math.log((v/totalWeight),2))
        entropy = entropy*-1

        return entropy

    def genSubsetForAttrValue(self,data,attr):
        subsets = {}
        posValues = []
        for line in data:
            line = line.strip().split("\t")
            if not posValues.__contains__(line[attr]):
                posValues.append(line[attr])

        for val in posValues:
            newSet = []
            for line in data:
                temp = line.strip().split("\t")
                if(temp[attr]==val):
                    newSet.append(line)
            subsets[val] = newSet

        return subsets

    def calculateSumOfWeights(self,data):
        totalWeight = 0.0
        for line in data:
            line = line.strip().split("\t")
            totalWeight += float(line[len(line)-1])

        return totalWeight

    def calculateInformationGain(self,data,subsets):
        infoGain = self.calculateEntropy(self.trainingData)
        allPointsInSubset = []
        for k,v in subsets.items():
            allPointsInSubset.extend(v)

        sampleWeight = self.calculateSumOfWeights(allPointsInSubset)

        for k,v in subsets.items():
            entropy = self.calculateEntropy(v)
            weightOfSubset = self.calculateSumOfWeights(v)
            infoGain = infoGain - ((weightOfSubset * entropy) / sampleWeight)

        return infoGain


    def generateDecisionStump(self,treeData,splitAttr):

        tree = {}
        for k,v in treeData.items():
            votedClass = self.majorityVotingInSubset(v)
            tree[k] = votedClass
        tree["splitAttr"]=splitAttr
        return tree

    def majorityVotingInSubset(self,data):
        #Identify the class
        classWeight = {}
        for line in data:
            line = line.strip().split("\t")
            className = line[0]
            classWeight.setdefault(className,0)
            classWeight[className] += float(line[len(line)-1])

        votedClass = max(classWeight.iteritems(), key=operator.itemgetter(1))[0]

        return votedClass

    def weakClassifierDecisionStump(self):
        treeData = {}
        prevGain = float("-inf")
        splitAttr = -1
        for i in range(1,len(self.trainingData[0].strip().split("\t"))-1,1):
            attr = i
            subsets = self.genSubsetForAttrValue(self.trainingData,attr)
            infoGain = self.calculateInformationGain(self.trainingData,subsets)

            if(infoGain > prevGain):
                prevGain = infoGain
                treeData = subsets
                splitAttr = i

        decisionStump = self.generateDecisionStump(treeData,splitAttr)
        return decisionStump


    def calculateAlpha(self,classifier):

        epsilonT = 0.0
        for line in self.trainingData:
            line = line.strip().split("\t")
            className = line[0]
            splitAttr = classifier["splitAttr"]
            attrVal = line[splitAttr]

            if not(classifier[attrVal]==className):
                epsilonT += float(line[len(line)-1])

        alphaT = 0.5 * math.log((1-epsilonT)/epsilonT,math.exp(1))

        return alphaT

    def updateWeightsofDataPoints(self,alphaT,Zt,classifier):
        output =os.linesep
        for i in range(0,len(self.trainingData)) :
            line = self.trainingData[i]
            line=line.strip().split("\t")
            weight = float(line[len(line)-1])
            splitAttr = classifier["splitAttr"]
            attrVal = line[splitAttr]
            yiht_xi = None
            if(classifier[attrVal]==line[0]):
                yiht_xi = 1
            else:
                yiht_xi = -1

            newWeight = (weight * math.exp(-1*alphaT*yiht_xi)) / Zt
            #update the weight
            line[len(line)-1] = str(newWeight)
            #update the data point in training data with new weights for next iteration classifier
            self.trainingData[i] = "\t".join(line)


    def adaboostAlgo(self):
        hypothesis = {}

        for i in range(0,self.boostItr):
            decisionStump = self.weakClassifierDecisionStump()
            # print(decisionStump)
            alpha = self.calculateAlpha(decisionStump)
            #Zt = self.calculateZt(decisionStump,alpha)
            Zt = self.calculateSumOfWeights(self.trainingData)
            output = self.updateWeightsofDataPoints(alpha,Zt,decisionStump)
            hypothesis[i] ={"alpha":alpha , "classifier": decisionStump }
            #print(str(decisionStump["splitAttr"]))

        self.runOnTestData(hypothesis)

        for k,v in hypothesis.items():
            print(str(v["alpha"]))

        # print(weightsData)


    def runOnTestData(self, hypothesis):
        totalSize = len(self.testingData)
        errorCount = 0

        for line in self.testingData:
            line = line.strip().split("\t")
            className = line[0]
            classWeight = {}
            for k,v in hypothesis.items():
                alpha = v["alpha"]
                classifier = v["classifier"]
                splitAttr = classifier["splitAttr"]
                dataValue = line[splitAttr]
                predClass = classifier[dataValue]
                classWeight.setdefault(predClass,0.0)
                classWeight[predClass]+=alpha
            votedClass = max(classWeight.items(), key=operator.itemgetter(1))[0]

            if not className == votedClass:
                errorCount+=1
        accuracy = float(totalSize - errorCount)/totalSize
        print(str(accuracy*100))


if __name__=="__main__":
    boostItr = int(sys.argv[1])
    trainingFile = sys.argv[2]
    testFile = sys.argv[2]
    adb = adaboost(trainingFile,testFile,boostItr)
    adb.adaboostAlgo()


