import numpy as np
import pandas as pd
import os
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import classification_report, confusion_matrix
from typing import Union
from scipy.spatial.distance import mahalanobis
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
TRAIN_FILE_PATH = os.path.join(FILE_DIR, "..", "baza_danych", "avila-tr.txt")
TEST_FILE_PATH = os.path.join(FILE_DIR, "..", "baza_danych", "avila-ts.txt")
COLUMN_NAMES = ["Intercolumnar distance", "Upper margin", "Lower margin", "Exploitation", "Row number", "Modular ratio", "Interlinear spacing", "Weight", "Peak number", "Modular ratio/Interlinear spacing", "Class"]
CLASS_NAMES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "W", "X", "Y"]
ATTRIBUTE_NAMES = ["Intercolumnar distance", "Upper margin", "Lower margin", "Exploitation", "Row number", "Modular ratio", "Interlinear spacing", "Weight", "Peak number", "Modular ratio/Interlinear spacing"]



class Dataset():
    def __init__(self, train_file:str, test_file:str, limit_to_classes:list[str] = CLASS_NAMES, limit_to_attributes:Union[list,int] = ATTRIBUTE_NAMES, train_test_split_ratio:float=None, normalize:bool=False):
        """Klasa pomocnicza do obsługi zbioru

        Args:
            train_file (str): Ścieżka do zbioru treningowego.
            test_file (str): Ścieżka do zbioru testowego.
            limit_to_classes (list[str], optional): Ograniczenie klas, lista klas które mają być brane pod uwagę.
                Próbki spoza tej listy zostaną usuniętę zarówno w zbiorze testowym jak i treningowym. Domyślnie CLASS_NAMES(wszystkie klasy).
            limit_to_attributes (Union[list,int], optional): Ograniczenie cech. Lista nazw cech do wzięcia pod uwagę. 
                Możliwe jest także podanie liczby całkowitej n w wyniku czego automatycznie wybrane zostanie n najlepszych cech korzystając z sklearn SelectKBest.
                Domyślnie ATTRIBUTE_NAMES(wszystkie cechy).
            train_test_split_ratio (float, optional): Ułamek według którego podzielić połączony oryginalny zbiór treningowy na testowy, do testowania skuteczności algorytmów.
                Domyślnie None umieszcza w self.train cały zbiór treningowy i w self.test cały zbiór testowy
            normalize (bool, optional): Czy normalizować za pomocą min-max próbki. 
                Próbki znormalizowane są Z-normalizacją, więc prawdopodobnie nie jest to konieczne. Domyślnie False.

        Raises:
            Exception: [description]
        """
        # wczytuje zbiory z .csv
        self.train = pd.read_csv(train_file, header=None)
        self.test = pd.read_csv(test_file, header=None)
        print(f"Loaded {self.train.shape} train dataset from file {train_file}.")
        print(f"Loaded {self.test.shape} test dataset from file {test_file}.")

        # łączy zbiory testowy i treningowy w jeden jeśli mają one być przeliczone
        self.combined = pd.concat((self.train, self.test), axis=0) 
        self.combined.reset_index(drop=True, inplace=True) # reset indeksowania
        print(f"Combined test and train into {self.combined.shape}.")
        combined_len = self.combined.shape[0]

        # jeśli split_ratio podano należy przetasować zbiory i podzielić na nowo
        if(train_test_split_ratio is not None):
            if(train_test_split_ratio >= 1 or train_test_split_ratio <= 0): raise Exception("Split ratio must be between (0,1).")
            # punkt podziału zbiorów na treningowy i testowy
            # zbiory dzielimy i resetujemy indeksy
            split_point = int(self.train.shape[0]*train_test_split_ratio)
            self.train = self.train[:split_point]
            self.test = self.train[split_point:]
            self.train.reset_index(drop=True, inplace=True)
            self.test.reset_index(drop=True, inplace=True)
            print(f"Train set was split into {self.train.shape}. Test set was split into {self.test.shape}.")
        else:
            print("Preserving original test-train split.")

        # nadaj kolumnom nazwy
        self.combined.columns = COLUMN_NAMES
        self.train.columns = COLUMN_NAMES
        self.test.columns = COLUMN_NAMES

        if(normalize): 
            self.normalize()

        # usuń ze zbioru próbki z nieanalizowanych klas i pomijane atrybuty
        self.preserved_classes = limit_to_classes
        self.preserved_attributes = limit_to_attributes
        self.reduce_classes(limit_to_classes)
        self.reduce_attributes()
        
        # stwórz słownik z próbkami każdej z klas
        self.class_split()
        # policz ich centroidy
        self.compute_means()
        # policz ich macierze kowariancji
        self.compute_covariance_matrices()

    def normalize(self): # nie normalizuje self.combined
        maxes = self.combined.max(axis=0)
        mins = self.combined.min(axis=0)
        span = maxes[:-1] - mins[:-1] # wszystkie poza ostatnią kolumną, zawierającą litere klasy

        for col in range(self.train.shape[1] - 1): # -1 aby nie minmaxowac klasy
            self.train.iloc[:,col] = self.train.iloc[:,col].subtract(mins[col]).divide(span[col])
        for col in range(self.test.shape[1] - 1):
            self.test.iloc[:,col] = self.test.iloc[:,col].subtract(mins[col]).divide(span[col])

    def reduce_classes(self, preserved:list[str]):
        """Odrzuca wszystkie próbki niepochodzące z klas podanych w liście preserved

        Args:
            preserved (lst[str]): lista klas do zachowania.
        """
        self.combined = self.combined[self.combined["Class"].isin(preserved)]
        self.train = self.train[self.train["Class"].isin(preserved)]
        self.test = self.test[self.test["Class"].isin(preserved)]
        self.combined.reset_index(drop=True, inplace=True)
        self.train.reset_index(drop=True, inplace=True)
        self.test.reset_index(drop=True, inplace=True)

    def reduce_attributes(self):
        """ Usuwa cechy(kolumny) niezawarte w self.preserved_attributes ze zbiorów testowego i treningowego
            Jeśli self.preserved_attributes jest liczbą to wyznacza najlepsze cechy za pomocą SelectKBest
        """
        if(type(self.preserved_attributes) is int):
            # <------------------------------>
            selector = SelectKBest(f_classif, k=self.preserved_attributes) # <-------------------- jak działa f_classif
            # <------------------------------>
            attrs_new = selector.fit_transform(self.train.iloc[:,:-1], self.train.iloc[:,-1])
            self.preserved_attributes = self.train.columns[:-1][selector.get_support()]
            print(f"Reduced attributes to {self.preserved_attributes} using SelectKBest")
            #self.train = attrs_new.concat(self.train.iloc[:,-1], axis=1)
            pass
        to_drop = [attr for attr in ATTRIBUTE_NAMES if attr not in self.preserved_attributes]
        self.combined.drop(to_drop, axis=1, inplace=True)
        self.train.drop(to_drop, axis=1, inplace=True)
        self.test.drop(to_drop, axis=1, inplace=True)
    
    def class_split(self):
        """Tworzy słownik dzielący próbki ze zbioru treningowego na klasy
            klasa:próbki
        """
        self.classed = {}
        for cl in self.preserved_classes:
            self.classed[cl] = self.train[self.train["Class"] == cl]
            self.classed[cl].reset_index(drop=True, inplace=True)
        for key in self.classed:
            print(f"{key}:{self.classed[key].shape[0]}")


    def compute_means(self):
        """Liczy centroidy próbek z klas. Wymaga podzielenia na słownik za pomocą self.class_split()
        """
        self.means = {}
        for cl in self.preserved_classes:
            self.means[cl] = self.classed[cl].mean(axis=0)

    def compute_covariance_matrices(self):
        """Liczy macierze kowariancji i ich odwrotności dla klas ze słownika. Wymaga podzielenia na słownik za pomocą self.class_split()
        """
        self.covariances = {}
        self.icovariances = {}
        for cl in self.preserved_classes:
            self.covariances[cl] = np.cov(self.classed[cl].iloc[:,:-1].to_numpy(), rowvar=False) # do not use the class column
            self.icovariances[cl] = np.linalg.inv(self.covariances[cl])

    def predict_test_samples(self, method, k=1):
        """Dokonuje predykcji klas ze zbioru self.test wykorzystując podaną metodę i parametry.

        Args:
            method (str): Metoda klasyfikacji, "knn", "nm" lub "mah-hm"
            k (int, optional): Wartość k do wykorzystania w kNN. Domyślnie 1.

        Returns:
            true, pred: zwraca jednokolumnowy dataframe z faktycznymi klasami i dataframe z odgadniętymi klasami
        """
        if(method == "knn"):
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(self.train.iloc[:,:-1], self.train.iloc[:,-1]) # [:,:-1] -> wszystkie kolumny(cechy) poza ostatnim(klasa), [:, -1] -> tylko klasa próbki
            predictions = model.predict(self.test.iloc[:,:-1])
        elif(method == "nm"):
            model = NearestCentroid(metric="euclidean")
            model.fit(self.train.iloc[:,:-1], self.train.iloc[:,-1])
            predictions = model.predict(self.test.iloc[:,:-1])
        elif(method == "mah-nm"):
            predictions = self.maha_nm(self.test.iloc[:,:-1])
        return (self.test.iloc[:,-1], predictions)

    def maha_nm(self, test):
        predictions = np.empty((test.shape[0],), dtype=object)

        for i in range(test.shape[0]): # dla kazdej probki z testowego
            class_assign = None
            min_dist = np.inf
            x = test.iloc[i].to_numpy() # stwórz z niej macierz numpy
            for c in self.classed:
                dist = mahalanobis(x, self.means[c].to_numpy(), self.icovariances[c])
                if(dist < min_dist):
                    class_assign = c
                    min_dist = dist
            predictions[i] = class_assign

        return predictions
# Do testowania algorytmów zbiory są podzielone na testowy i treningowy za pomocą tego współczynnika; 
# testowy bedzie zawierał PARAM_SPLIT_TEST próbek oryginalego zbioru testowego;
# PARAM_SPLIT_TEST może także przyjąć wartość None sugerując użycie oryginalnego podziału na test i train;
PARAM_SPLIT_TEST = None # None 

ds = Dataset(
    TRAIN_FILE_PATH, 
    TEST_FILE_PATH, 
    limit_to_classes = ["D", "G", "H", "X"], 
    limit_to_attributes = 5,
    train_test_split_ratio=PARAM_SPLIT_TEST, 
    normalize=False
)
true, pred = ds.predict_test_samples(method="nm")
print("-------------- NM ---------------")
print(classification_report(true, pred))

true, pred = ds.predict_test_samples(method="mah-nm")
print("-------------- NM-Mah ---------------")
print(classification_report(true, pred))

true, pred = ds.predict_test_samples(method="knn", k=1)
print("-------------- 1-NN ---------------")
print(classification_report(true, pred))

true, pred = ds.predict_test_samples(method="knn", k=3)
print("-------------- 3-NN ---------------")
print(classification_report(true, pred))

true, pred = ds.predict_test_samples(method="knn", k=5)
print("-------------- 5-NN ---------------")
print(classification_report(true, pred))

true, pred = ds.predict_test_samples(method="knn", k=9)
print("-------------- 9-NN ---------------")
print(classification_report(true, pred))
