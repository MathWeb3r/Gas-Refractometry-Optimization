import json
import numpy as np
from pathlib import Path

# espera essa estruta de arquivo
#
# gas_refractometry_optimize
# ├── gasopt
# │   ├── __init__.py
# │   ├── dispersion.py
# ├── data
# │   └── gas_database.json

# Caminho do arquivo atual
current_file = Path(__file__).resolve()

# .parent é 'scripts/', .parent.parent é a raiz do projeto 'gas_refractometry_optimize/'
BASE_DIR = current_file.parent.parent 

# caminho do database
DATAPATH = BASE_DIR / 'data' / 'gas_database.json'


# Essa classe será a responsável por tratar a dispersão dos gases
# espero que ela me dar valores de refratividade por comprimento de onda
# e manipular todos os tipos de valores nesse assunto
class Sellmeier():
    def __init__(self, A: float, B: np.ndarray , C: np.ndarray):
        '''
        A, B and C are the sellmeier equation coefficients
        B and C are arrays wich each item is Bi and Ci, respectively
        A is the linar a coefficient, a float number

        Parameters:
        ----------

        *A: float, liner coefficients
        *B: array_like with Bi coefficients
        *C: array_like with Ci coefficients
        '''
        
        self.A = A
        self.B = np.asarray(B)
        self.C = np.asarray(C)
    
    def calculate_sellmeier(self, x):
        '''
        This method calculate the sellmeier equation 
        for a wavelengh x, in micrometers

        Parameters:
        ----------

        *x: float, wavelenght in micrometers (μm)
        '''
        
        # performing the sum
        s = 0

        stack = np.stack((self.B, self.C), axis=1)

        for b, c in stack:
            #print(f' + ({b}/{c} - x^-2)', end='')
            s += b / (c - x**-2)

        # summing the linear term
        # the result is (n-1), or refractivity
        n_1 = self.A + s
        return n_1

    def dispersion(self, start=400, end=700, n=None) -> np.array:
        '''
        this method returns the dispersion in a wavelenght spcetrum
        basically, calculate de sellmeir for each wavelenght

        Parameters:
        ----------
        *start: float, initial point in micrometers (μm)
        *end: float, end point in micrometers (μm)
        *n: int, number of points in the range
        '''

        if n == None:
            x = np.linspace(start, end)
        else:
            x = np.linspace(start, end, n)

        y = np.array([self.calculate_sellmeier(xi) for xi in x])

        # if y.ndim > 1 it means that it contains [refractivity, sigma]
        # and the last dimension is the sigma
        if y.ndim > 1:
            refractivity = y[:, 0]
            sigma = y[:, 1]

            return np.stack((refractivity, sigma), axis=1)
        
        else:
            return np.array([y, 0])


class Gas(Sellmeier):
    def __init__(self, id = None, name = None, A = None, B = None , C = None, 
                 spc_range = None, t=273.15, p=101325, magnitude=1E-8):
        '''
        If id is guiven, it will get the data from
        the gas_database.json, otherwise, it will be 
        created with the given parameters.

        Parameters:
        ----------

        *name: str, latex name that will be shown in plots
        *t: float, reference temperature in Kelvin
        *p: float, reference pressure in Pascal
        *A: float, liner sellmeier coefficients
        *B: array_like with Bi sellmeier coefficients
        *C: array_like with Ci sellmeier coefficients
        '''
        if id != None:
            self._loadData(id)
        else:
            super().__init__(A=A, B=B, C=C)
            self.name = name
            self.t_ref = t
            self.p_ref = p
            self.range = spc_range
            self.multiplier = magnitude

    def _loadData(self, id):
        '''
        This method load the data from the gas_database.json
        and set the attributes of the class

        Parameters:
        ----------
        *id: str, gas id in the database
        '''
        with open(DATAPATH, 'r') as f:
            data = json.load(f)

            # sellmeier coefficients
            A = data[id]['coef']['A']       
            B = data[id]['coef']['B']
            C = data[id]['coef']['C']
            super().__init__(A=A, B=B, C=C)

            # metadata
            self.name = data[id]['metadata']['gas']
            self.t_ref = data[id]['metadata']['t_ref']
            self.p_ref = data[id]['metadata']['p_ref']
            self.multiplier = data[id]['metadata']['multiplier']
            
            # intervalo de validade dos coeficientes
            inf = data[id]['spectral_range_nm']['min']
            sup = data[id]['spectral_range_nm']['max']
            self.range = (inf, sup)

            # artigo de referencia
            self.reference = data[id]['metadata']['reference']

            # incerteza
            self.sigma = data[id]['sigma']

    def __repr__(self):
        return f'{self.name} de {self.reference}'

    def calculate_sellmeier(self, x):
        """
        this method calculate the sellmeier equation 
        for a wavelenght x, in micrometers, 
        and multiply the result by the gas multiplier

        Parameters:
        ----------

        *x: float, wavelenght in micrometers (μm)
        
        Returns:
        -------
        *np.ndarray: array with [refractivity, sigma]
                     multiplied by the gas multiplier
        """
        # add coeficiente de temperatura e pressão\
        refractivity = super().calculate_sellmeier(x)
        return np.asarray([refractivity, self.sigma]) * self.multiplier
    
    def dispersion(self, start=None, end=None, n=None):
        # add coeficiente de temperatura e pressão
        if start == None and end == None:
            start, end = self.range[0], self.range[1]

        return super().dispersion(start, end, n)

if __name__ == '__main__':
    gas = Gas('co2_1')
    y = gas.dispersion()
    x = np.linspace(gas.range[0], gas.range[1], len(y))
