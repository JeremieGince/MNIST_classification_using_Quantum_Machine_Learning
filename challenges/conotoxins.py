import torch
import pandas as pd
import os
import numpy as np


class CharsInAllData:
    def __init__(self, trainingData, testData):
        self.__allData = np.append(trainingData, testData)
        self.__oneHots, self.nbLetters = self.__initializeOneHotVectors()

    def __getNumberOfUniqueLetters(self):
        s = set()
        for code in self.__allData:
            s = s.union(set(code))
        return s

    def __initializeOneHotVectors(self):
        s = sorted(self.__getNumberOfUniqueLetters())
        nbLetters = len(s)
        oneHotVectors = {}
        for index, letter in enumerate(s):
            oneHot = np.zeros(nbLetters)
            oneHot[index] = 1
            oneHotVectors[letter] = oneHot
        return oneHotVectors, nbLetters

    def getOneHotFromChar(self, char: str):
        return self.__oneHots[char]

    @property
    def conversionDict(self):
        return self.__oneHots.copy()


class TextDataToMatrix:

    def __init__(self, data, charToOneHotObj: CharsInAllData, maxDim: int):
        self.__data = data
        self.__preparedData = self.__prepareData(charToOneHotObj, maxDim)

    def __prepareData(self, convertObj: CharsInAllData, maxDim: int) -> np.ndarray:
        fullTensor = []
        for d in self.__data:
            matrixData = np.zeros((convertObj.nbLetters, maxDim), dtype=int)
            for index, c in enumerate(d):
                oneHot = convertObj.getOneHotFromChar(c)
                matrixData[:, index] = oneHot
            fullTensor.append(matrixData)
        return np.dstack(fullTensor)

    @property
    def numericalTensor(self):
        return self.__preparedData.copy()


class LSTM_Model:

    def __init__(self, trainData: np.ndarray, testData: np.ndarray, hiddenSize: int, nbLayers: int, bias: bool = False,
                 batchFirst: bool = False, dropout: int = 0, bidirectional: bool = False):
        trainDataShape = trainData.shape
        nbFeatures = trainDataShape[0] * trainDataShape[1]
        self.LSTM = torch.nn.LSTM(input_size=nbFeatures, hidden_size=hiddenSize, num_layers=nbLayers, bias=bias,
                                  batch_first=batchFirst, dropout=dropout, bidirectional=bidirectional)
        self.trainData = trainData
        self.testData = testData
        self.__nbLayers = nbLayers
        self.__nbDirections = int(bidirectional) + 1
        self.__hiddenSize = hiddenSize

    def input(self, h0: np.ndarray = None, c0: np.ndarray = None, batchSize: int = 5):
        if h0 is None:
            h0 = np.random.rand(self.__nbLayers * self.__nbDirections, batchSize, self.__hiddenSize)
        if c0 is None:
            c0 = np.random.rand(self.__nbLayers * self.__nbDirections, batchSize, self.__hiddenSize)
        input_ = self.trainData[]
        self.LSTM.train()


if __name__ == '__main__':
    path = os.getcwd()
    path = os.path.join(path, "data", "conotoxins.xls")

    data = pd.read_excel(path, [0, 1])
    trainingData = data[0].SEQUENCE
    testData = data[1].SEQUENCE
    maxLen = max(max([len(d) for d in trainingData]), max([len(d) for d in testData]))
    # maxLen *= 1
    print(maxLen)
    convert = CharsInAllData(trainingData, testData)
    t = TextDataToMatrix(trainingData, convert, int(maxLen))
    numericalData = t.numericalTensor
