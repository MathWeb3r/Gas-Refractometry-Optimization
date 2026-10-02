import itertools
import numpy as np
import multiprocessing
from gasopt import GasMix
from math import comb

from heapq import heapify
from heapq import heappush as push
from heapq import heappop as hpop

# This class calculates the the kappa for every possible combination
# of elements of a sample. Then find a N size smallest kappas set   

# Para não estorar o limite de memória, ou utilizar memória 
# demais (deixando o pc lerdo), 


# Para manter o consumo constante de memória vou salvar apenas as
# N melhores combinações em um max heap. Tambem vou dividir as 
# combinações em pedaços (chunks).
# Então vou calcular os kappas em um chunk, 


class GridSearch():
    def __init__(self, sample: np.ndarray = np.array([]), k: int = 4):
        '''
        Params:
        sample: np.array, sample space for the combinations, ie
                set of wavelenghts in micrometers 
        k: int, size of each combination 
        '''
        
        #
        self.sample = sample
        self.k = k

        # estou contando com o fato que o na minha bases os 
        # gases são salvos com um id que é o nome + indice
        # ex: co2_1, co2_2
        # entao vou salvar o gases sem o indice, e escolher o 
        # indice depois, baseado na validade de cada equação
        #
        # depois pensar em como o usuario escolhe isso
        self.gasesid = ['co2_1', 'o2_1', 'ar_1', 'n2_1']

        # temperatura da mistura
        self.temp = 273.15

        # tamanho do grid_search
        # n escolhe k, (n k)
        self.size = comb(len(self.sample), self.k)

        # inicializa um iteravel com as combinações
        # as combinações serão feitas com os índeices 
        # de sample
        idx = range(len(self.sample))
        self.combs = itertools.combinations(idx, self.k)

        # max heap com as melhores 
        self.heap = []
        heapify(self.heap)

    def __len__(self):
        return self.size

    def _get_chunck(self, size: int = 0):
        '''
        Escolhe um conjunto de tamanho size de combinações ainda não testadas

        Se restarem menos combinações que o tamanho size, retorn apenas as 
        as combinações que faltam  
        '''
        return list(itertools.islice(self.combs, size))

    def _get_bigmatrix(self):
        '''
        Calucula de antemão todas as refratividades. Isso é, refratividade
        para todos os comprimentos de onda e todos os gases.

        todo: fazer um verficação de validade da equações por gas. Por exemplo,
        se o comprimento de onda for menor que 740nm não usar o coeficientes de 
        'co2_1' e usar os de 'cos_2'.
        '''

        self.big_matrix = GasMix(
            ids=self.gasesid,
            wvlens=np.array(self.sample), # converte nm p/ um
            temp=self.temp
        ).get_matrix()

    def _calculate_kappa(self, chunk):
        '''
        
        '''
        # cria uma matrix M x N
        # M: Numero de gases 
        # N: Numero de lasers
        self._get_bigmatrix()

        # tensor tamanho: M x N x chunksize
        tensor = self.bigmatrix[chunk]
        
        # tensor dos valores singulares
        s = np.linalg.svd(tensor, compute_uv=False)
        
        # o numero de condição pode ser definido como a razao
        # k(A) = M/m
        # M: max que A estica um vetor x
        # m: min que A comprime um vetor x
        # Ao mesmo tempo, isso também é exatamente a razão entre
        # o menor valor singular de A com o maior
        kappas = s[:, 0] / s[:, -1]
        return kappas

    def _update_heap(self, kappas):
        '''
        Recebe uma lista de candidatos kappas e verifica se eles são
        bons ou ruins.

        Kappas menores que o maior elemento do heap são salvos, e 
        kappas menores esquecidos.

        O heap é mantido com tamanho fixo
        '''

        for k in kappas:
            # enche o heap até o tamanho 
            if len(self.heap) < self.nbest:
                push(self.heap, -k)
            else:
                # O heapq é implementado como um min heap
                # então estou usando os opostos (*-1)
                if -k > -self.heap[0]:
                    hpop(self.heap)
                    push(self.heap, k)


    def run_seach(self, n_best: int = 100, chunk_size: int = 1000):
        '''

        n_best: int, numero do conjunto das melhores combinações
        '''
        
        self.nbest = n_best

        for i in range(self.size // chunk_size + 1):
            chunk = self._get_chunck(chunk_size)
            kappas = self._calculate_kappa(chunk)
            self._update_heap(kappas)            

        return list(self.heap)

if __name__ == '__main__':
    a = ['a', 'b', 'c', 'd', 'e']
    
    c = GridSearch(sample=a, k=3)
    
    print(c.run_seach())
    print('N combinacaoes: ', len(c))