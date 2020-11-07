import numpy as np
import requests
import re
import matplotlib.pyplot as plt


class DataDownloader:

    def __init__(self, url: str):
        """
        Constructor of the DataDownloader class. We download a text file from internet. This class is not used if the
        data is already on the computer.
        :param url: The url where the data is stored.
        """
        self.url = url
        # Here, we use the requests module to get the text from the webpage.
        self.data = requests.get(self.url).text

    def saveToTxtFile(self, filename: str):
        """
        Method that saves the downloaded data into a text file. This file is then saved on the computer.
        :param filename: The name of the file where the data is stored.
        """
        # Here, we open the file with open. If filename doesn't exists, we create it. Otherwise, we will overwrite what
        # is already there. The argument "w" means that we want to write stuff.
        with open(filename, "w") as f:
            f.write(self.data)


class SpectralLinesFinder:

    def __init__(self, filename: str):
        """
        Constructor of the main class used to find spectral lines from a specific spectrum.
        :param filename: The name of the file where the data is saved. Must be in ascii format.
        """
        # Here, we open the file where the data is. The argument "r" means that we want to read the file.
        with open(filename, "r") as f:
            self.data = f.read()

        # Here, we split the content for each line in the spectrum file.
        dataLines = self.data.split("\n")

        # Here, we access the object information (i.e. name). Using pop allow us to access the line and remove it since
        # it is not useful anymore.
        self.object = dataLines.pop(0)

        # Here, we access the file information (number of lines and columns). Like for the previous line, we pop it.
        self.fileInfo = dataLines.pop(0)

        # If the last line is nothing but whitespaces, we get rid of it. The strip method is used to remove whitespaces,
        # so if it is only whitespaces, it will be empty after the strip.
        if dataLines[-1].strip() == "":
            del dataLines[-1]

        # Here, we remove the first whitespaces (if there are any) at the beginning of each line.
        dataLines = [re.sub(r"^\s*", "", line) for line in dataLines]

        # Here, we replace whitespaces between columns with a single comma. It will be easier to manipulate!
        dataLines = [re.sub(r"[\s]+", ",", line) for line in dataLines]

        # Here, we take our lines and split them again into columns. That way, we have a 2D array (hence why we create
        # an array using the module NumPy). The argument dtype is used to specify what is the type of the data
        # (here, it is float numbers). We transpose to have the wavelength as the first row and the intensity as the
        # second row. The other rows will be removed.
        self.data = np.array([line.split(",") for line in dataLines], dtype=float).T

        # Here, we keep only the first two rows (or the first two columns of the original dataset).
        self.data = self.data[:2]

    def findSpectralLines(self, wavelengthRange: tuple, emissionThreshold: float, absorptionThreshold: float,
                          continuum: float = 1):
        """
        Method used to find spectral lines.
        :param wavelengthRange: Tuple of two values: the start of the range and the end of the range. This parameter
        can be None if one wants to consider all the spectrum.
        :param emissionThreshold: Float representing the minimum intensity (depending on the continuum)
        considered a proper spectra line. For example, if we give 0.5, the minimum will be at 0.5*continuum.
        :param absorptionThreshold: Float representing the maximum intensity (depending on the continuum)
        considered a proper spectra line. For example, if we give 0.5, the maximum will be at 0.5*continuum.
        :param continuum: Float representing the continuum line. By default, it is 1.
        :return: An array containing
        """
        if wavelengthRange is None:
            wavelengthRange = (0, self.data.shape[1])
        usefulData = self.data[:, :wavelengthRange[0], wavelengthRange[1]]

    def showSpectrum(self, wavelengthRange: tuple = None):
        if wavelengthRange is None:
            wavelengthRange = (0, self.data.shape[1])
        usedData = self.data[:, :wavelengthRange[0], wavelengthRange[1]]
        plt.scatter(*usedData, marker=".", s=2)
        plt.show()

    @staticmethod
    def findValuesAbove(data: np.ndarray, minimum: float):
        pass


if __name__ == '__main__':
    # url_1 = r"https://www.cfht.hawaii.edu/~manset/Hackathon2020/1835518in.s"
    # url_2 = r"https://www.cfht.hawaii.edu/~manset/Hackathon2020/2107396pn.s"
    fname1 = "test.txt"
    fname2 = "test2.txt"
    # DataDownloader(url_1).saveToTxtFile(fname1)
    # DataDownloader(url_2).saveToTxtFile(fname2)
    lines = SpectralLinesFinder(fname1)
    lines.showSpectrum()
