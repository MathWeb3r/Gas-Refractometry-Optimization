import numpy as np
from numpy.linalg import cond

# fix this 
# when debugging from this file (__name__ == '__main'__)
# only dispersion works. when running in a import works
# with .dispersion

from gasopt.dispersion import Gas
from gasopt.database import GasDatabase

DB = GasDatabase()           

class GasMix():
    """
    Classe que representa uma mistura de gases para cálculo da matriz de refratividade
    usando a regra de Gladstone-Dale.
    """
    def __init__(self, gases: list = None, wvlens: list = None, temp: float = None, 
                ids: list = None):
        """
        Inicializa a mistura de gases.

        Args:
            gases (list): Uma lista de objetos gasopt.Gas.
            ids (list): Uma lista de id dos gases no banco de dados.
            wvlens (list): Comprimentos de onda em micrômetros.
            temp (float): Temperatura da mistura em Kelvin.
        """

        if gases == None and ids == None:
            raise Exception('No gases or IDs provided')

        if gases != None and ids != None:
            raise Exception('Only one of the arguments "gases" or "ids" can be provided')

        # ids != None é o caso em que o usuário passou uma lista para
        # query no banco de dados. Que pode ser genérica ou não
        if ids != None:
            # valida todas as entradas
            ids = DB.is_valid_ids_list(ids) 
            self.ids = ids

            self.is_generic = DB.all_generic(ids)

            if self.is_generic:
                self.gases = None

            else:
                self.gases = np.array([Gas(id) for id in ids])
 
        else:
            self.is_generic = False
            self.gases = np.array(gases)
            self.ids = [g.id for g in self.gases]
        
        # valida os comprimentos de onda
        DB.validate_wavelengths(wvlens)
        self.wvlens  = np.array(wvlens)

        if temp == None or temp <= 0:
            raise ValueError('Temperature must be not Null and Positive.')
        self.temp = temp

        #[print(g.name) for g in self.gases]

    def _construct_best_matrix(self, ids, wvlens):
        '''
        Constroi a matrix A com as melhores equações para cada elemento Aij
        '''
        pass

    def get_fullmatrix(self):
        '''

        '''

        if self.is_generic:

            # mtx tem 3 dimensões
            # i -> comprimento de ondas
            # j -> gases 
            # k -> valor e sigma
            n_wv = len(self.wvlens)
            n_gases = len(self.ids)
            mtx = np.zeros((n_wv, n_gases, 2))
            
            # track gases ids
            gas_mtx = np.empty((n_wv, n_gases), dtype=object)
            
            for i, wv in enumerate(self.wvlens):
                for j, gasid in enumerate(self.ids):
                    best_gas = DB.get_best_equation(gas_name=gasid, wvlen=wv)
                    mtx[i][j] = self.get_rafrac(best_gas, wv)
                    gas_mtx[i][j] = best_gas.id

            self.gases_id = gas_mtx

        else:
            mtx = []
            for wv in self.wvlens:
                refracs = [self.get_rafrac(g, wv) for g in self.gases]
                mtx.append(refracs)
        
        res = MixtureMatrix(mtx)
        res.gases = self.ids
        res.wvlens = self.wvlens
        return res
        
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
        self.gases = getattr(obj, 'gases', None)
        self.wvlens = getattr(obj, 'wvlens', None)

    def cond_number(self):
        """
        This method returns the conditional number of the matrix
        """
        if self.ndim == 3 and self.shape[2] == 2:
            return cond(self[:, :, 0])
        return cond(self)

    def __repr__(self):
        parts = [f"shape={self.shape}"]
        
        gases = getattr(self, 'gases', None)
        if gases is not None:
            parts.append(f"gases={list(gases)}")

        wvlens = getattr(self, 'wvlens', None)
        if wvlens is not None:
            try:
                parts.append(f"wvlens={[round(float(w), 4) for w in wvlens]}")
            except Exception:
                parts.append(f"wvlens={list(wvlens)}")

        if (self.ndim == 2 or (self.ndim == 3 and self.shape[2] == 2)) and min(self.shape[:2]) > 0:
            try:
                k = self.cond_number()
                if np.isinf(k):
                    parts.append("cond=inf")
                elif k >= 1e4:
                    parts.append(f"cond={k:.2e}")
                else:
                    parts.append(f"cond={k:.2f}")
            except Exception:
                pass

        header = f"<MixtureMatrix: {', '.join(parts)}>"
        arr_str = np.array2string(np.asarray(self))
        return f"{header}\n{arr_str}"

if __name__ == '__main__':
    operator = GasMix(
        ids=['co2', 'n2'],
        wvlens=[0.532, 0.6328, 0.780, 1.064],
        temp=273
    )

    print('\nFull Matrix')
    print(operator.get_fullmatrix())
    print('\nMatrix')
    print(operator.get_matrix())
    print('\nSigma Matrix')
    print(operator.get_sigma_matrix())
    print('\nEquações Selecionadas (Rastreabilidade):')
    print(operator.gases_id)

    