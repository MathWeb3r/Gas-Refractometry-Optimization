import numpy as np
from numpy.linalg import cond

# fix this 
# when debugging from this file (__name__ == '__main'__)
# only dispersion works. when running in a import works
# with .dispersion

from gasopt.dispersion import Gas

class GasMix():
    """
    Classe que representa uma mistura de gases para cálculo da matriz de refratividade
    usando a regra de Gladstone-Dale.
    """
    def __init__(self, gases: list = [], wvlens: list = None, temp: float = None, ids: list = []):
        """
        Inicializa a mistura de gases.

        Args:
            gases (list): Uma lista de objetos gasopt.Gas.
            wvlens (list): Comprimentos de onda em micrômetros.
            temp (float): Temperatura da mistura em Kelvin.
        """

        if gases == [] and ids == []:
            raise Exception('No gases or IDs provided')

        if gases != [] and ids != []:
            raise Exception('Only one of the arguments "gases" or "ids" can be provided')

        if ids != []:
            try:
                self.gases = np.array([Gas(id) for id in ids])
            except:
                raise Exception(f'Invalid Gases IDs: {ids}')
        else:
            self.gases = np.array(gases)

        self.wvlens  = np.array(wvlens)
        self.temp = temp

        #[print(g.name) for g in self.gases]

    def get_fullmatrix(self):
        mtx = []
        for wv in self.wvlens:
            refracs = [self.get_rafrac(g, wv) for g in self.gases]
            mtx.append(refracs)

        return MixtureMatrix(mtx)
        
    def get_sigma_matrix(self) -> np.ndarray:
        mtx = self.get_fullmatrix()
        return mtx[:, :, 1]

    def get_matrix(self) -> np.ndarray:
        mtx = self.get_fullmatrix()
        return mtx[:, :, 0]
        
    def get_rafrac(self, gas: Gas, wvlen: float) -> np.array:
        '''
        Verificar esse metodo dps, poderia ser da classe Gas
        '''
        #return [g.sellmeier(wvlen)*g.magnitude for g in self.gases]
        
        # sanitazação dos dados
        wvlen = float(wvlen)

        # fator linear da temperatura
        t_factor = gas.t_ref/self.temp
        return gas.calculate_sellmeier(wvlen)*t_factor
        

class MixtureMatrix(np.ndarray):
    """
    Matrix from the mixture of gases, from the Gladstone-Dale refractive
    rule for gases. It's a A = (a)ij matrix, where i = lambda (wavelength)
    and j = refractivity of each gas. This class inherits numpy.matrix,
    following this numpy subclassing method 
    (https://numpy.org/doc/stable/user/basics.subclassing.html)

    But it must be call from a explicit construtcor, i.e. Mixture Matrix()
    because, initially, i will not implement support for casting from view
    or new-from-template.

    Parameters:
    ----------
    *mtx of gas and lambda 
    """
    def __new__(cls, input_array):
        #print('input_array', input_array)
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return 

    def cond_number(self):
        """
        This method returns the conditional number of the matrix
        """
        return cond(self)

if __name__ == '__main__':
    operator = GasMix(
        ids=['co2_1', 'n2_1'],
        wvlens=[100.0, 200.0],
        temp=273
    )

    print('\nFull Matrix')
    print(operator.get_fullmatrix())
    print('\nMatrix')
    print(operator.get_matrix())
    print('\nSigma Matrix')
    print(operator.get_sigma_matrix())

    