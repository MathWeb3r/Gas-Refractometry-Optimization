import numpy as np
import emcee as mc
import matplotlib.pyplot as plt

class MCMCInference:
    def __init__(self, mtx, sigma_mtx, phi_real=None, erro=None, b_obs=None, sigma=None):
        """
        Inicializa a classe de inferência Bayesiana MCMC.
        
        Args:
            mtx (numpy.ndarray): Matriz de refratividades.
            x_real (numpy.ndarray, opcional): Vetor x real (frações molares). 
            Usado para simular b_obs e sigma caso não sejam fornecidos diretamente.
            erro (float, opcional): Erro percentual para simulação de b_obs (ex: 0.001 para 0.1%).
            b_obs (numpy.ndarray, opcional): Vetor de refratividade observado.
            sigma (numpy.ndarray ou float, opcional): Erro associado às observações.
        """
        self.mtx = mtx
        self.sigma_mtx = sigma_mtx
        # o numero de dimensões é igual 
        # ao numero de gases 
        # menos 1
        self.ndim = mtx.shape[0] - 1
        
        if b_obs is not None and sigma is not None:
            self.b_obs = b_obs
            self.sigma = sigma
        elif phi_real is not None and erro is not None:
            # a refratividade real
            b_real = np.dot(mtx, phi_real)
            
            # adicionando um ruido baseado no erro
            self.sigma = erro * np.abs(b_real)
            self.b_obs = b_real + np.random.normal(loc=0, scale=self.sigma, size=b_real.shape)
        else:
            raise ValueError("É necessário fornecer b_obs e sigma, ou phi_real e erro para simulação.")

        self.sampler = None

    def log_prior(self, x):
        """
        x: vetor de soluções
        """
        #
        # O prior é p(phi)
        # é a probabilidade do meu vetor de solução ser real
        # aqui eu verifico o sentido físico desse vetor  
        #
        # return -inf <--> estado impossivel
        # pois log(0) = -inf
        # return 0 <--> estado possivel
        # pois log(1) = 0
        #
        
        # constrain do intervalo [0, 1]
        if np.any(x > 1) or np.any(x < 0):
            return -np.inf

        # constrain do elemento 
        # da dependecia linear
        lastx = 1.0 - np.sum(x)
        if lastx <= 0 or lastx >= 1:
            return -np.inf

        return 0.0

    def log_likelihood(self, x):
        """
        likelihood é P(b | phi)
        mede o quanto b produz o resultado o phi
        Supondo que a distribuição dos erros é gaussiana
        a função de custo é o Chi² do fitting
        Chi²(phi) = Sum (b_i - (A . phi))²/sigma²
        
         P(b|phi) ~ exp(-0.5 * Chi²)
         log P(b|phi) ~ -0.5 * Chi²
        """
        
        # reconstruindo o vetor completo
        lastx = 1 - sum(x)
        x_completo = np.append(x, lastx)
        x_completo = np.array(x_completo)

        # propagação de incerteza
        # a incerteza total é a soma das incertezas
        # instrumentais e da incerteza dos gases
        #
        # sigma2_total: incerteza total quadrada
        #
        # contribuição instrumental
        sigma2_total = self.sigma**2
        # contribuição dos gases
        sigma2_total += np.sum(np.dot(self.sigma_mtx, x_completo))**2

        # A refratividade real
        b_teorico = np.dot(self.mtx, x_completo)

        # Função de custo = chi
        chi2 = (self.b_obs - b_teorico)**2
        chi2 = chi2/sigma2_total
        chi2 = np.sum(chi2)    

        # log likelihood
        likelihood = -0.5*chi2

        return likelihood

    def log_prob(self, x):
        """
        a probabilidade p(phi|b) 
        é a prob de phi ser a solução correta, dado b
        p(phi|b) ~ p(b|phi) * p(phi)  
        log[p(phi|b)] ~ log[p(b|phi)] + log[p(phi)] 
        """
        # log[p(phi)]
        prior = self.log_prior(x)

        # log[p(b|phi)]
        likelihood = self.log_likelihood(x)

        # log[p(phi|b)]
        prob = prior + likelihood
        return prob

    def sequencia(self):
        """
        Gera uma sequência de valores aleatórios estritamente positivos (maiores que zero)
        para a inicialização dos walkers no MCMC.

        Amostra uniformemente um vetor de frações molares para os N gases (self.ndim + 1)
        no simplex utilizando a distribuição de Dirichlet (alpha = 1). Isso garante que:
        1. Nenhuma componente seja igual a zero (todas as frações são estritamente > 0).
        2. A soma de todas as N frações molares seja exatamente 1.0.
        3. A soma das primeiras `self.ndim` componentes seja estritamente menor que 1.0,
           deixando uma fração positiva para a última componente dependente (1 - sum(x) > 0).

        Returns:
            numpy.ndarray: Vetor unidimensional de tamanho `self.ndim` contendo frações 
            molares iniciais aleatórias e estritamente positivas.
        """
        # Amostra N frações molares (self.ndim + 1) no simplex (soma = 1.0, todas > 0, sem zeros)
        fractions = np.random.dirichlet(np.ones(self.ndim + 1))
        
        # Retorna apenas as primeiras (N - 1) componentes independentes
        # A última componente (garantidamente > 0) permanece implícita em (1 - sum(x))
        return fractions[:self.ndim]

    def gerar_p0(self, n_walkers):
        """
        função que gera uma posição inicial para
        os walkers, consistenete com a física
        do problema
        """
        p0 = np.zeros((n_walkers, self.ndim))
        for i in range(n_walkers):
            p0[i] = self.sequencia()

        return p0

    def get_sampler(self, n_walkers=32):
        """
        Cria e retorna o EnsembleSampler do emcee.
        """
        # criando o sampler
        sampler = mc.EnsembleSampler(
            nwalkers=n_walkers, 
            ndim=self.ndim, 
            log_prob_fn=self.log_prob
        )

        return sampler
        
    def run_mcmc(self, n_walkers: int = 32, n_steps: int = 7500, p0: np.array = None, progress: bool = False):
        """ 
        Cria o sampler, gera o ponto inicial (p0) e executa o MCMC.
        """
        # é preciso definir um ponto para cada walker
        # começar. Usaremos a posição real com um pequeno ruído
        # para garantir que comece em uma região válida (soma < 1 e valores entre 0 e 1)
        p0 = self.gerar_p0(n_walkers=n_walkers)
        
        sampler = self.get_sampler(n_walkers=n_walkers)
        self.sampler = sampler

        sampler.run_mcmc(
            initial_state=p0,
            nsteps=n_steps,
            progress=progress
        )
        
        return sampler
        
    def get_results(self, sampler = None, burn_in=300):
        """
        Extrai as cadeias de resultado a partir de um sampler percorrido.
        """

        if sampler == None and self.sampler == None:
            raise ValueError("No sampler provided. Run run_mcmc() or get_sampler() first.")
        
        elif sampler == None and self.sampler != None:
            sampler = self.sampler

        # cadeia walker a walker
        # espero veficar a convergencia
        chain = sampler.get_chain()
        
        # cadeia "achatada"
        # espero veficar a distribuição posterior
        chainflat = sampler.get_chain(discard=burn_in, flat=True)

        # Recupera o componente de referência x_k = 1 - sum(x_i)
        # pela dependência linear do fechamento do simplex.
        # Na cadeia completa (steps, walkers, ndim)
        lastx_chain = 1.0 - np.sum(chain, axis=-1, keepdims=True)
        chain = np.concatenate([chain, lastx_chain], axis=-1)
        
        # Na cadeia achatada (n_samples, ndim)
        lastx_flat = 1.0 - np.sum(chainflat, axis=-1, keepdims=True)
        chainflat = np.concatenate([chainflat, lastx_flat], axis=-1)
        
        return chain, chainflat

    def mean_interval(self, chainflat, confidence=0.95):
        """
        Cálculo da média e intervalo de credibilidade (bayesiano) para cada parâmetro.
        
        Args:
            chainflat (np.ndarray): Cadeia no formato achatado (n_samples, n_params).
            confidence (float): Nível de confiança do intervalo (padrão: 0.95 para 95%).
            
        Returns:
            tuple: (mean_vector, intervals_matrix)
                - mean_vector: Vetor 1D com a média de cada parâmetro.
                - intervals_matrix: Matriz 2D (n_params, 2) com [lower_bound, upper_bound] para cada parâmetro.
        """
        n_params = chainflat.shape[1]
        
        # 1. Calcular a média para cada parâmetro
        mean_vector = np.mean(chainflat, axis=0)
        
        # 2. Calcular os limites do intervalo de credibilidade
        # Para um intervalo de confiança de 95%, queremos os quantis 0.025 e 0.975
        lower_bound = (1.0 - confidence) / 2.0
        upper_bound = 1.0 - lower_bound
        
        intervals_matrix = np.zeros((n_params, 2))
        
        for i in range(n_params):
            param_chain = chainflat[:, i]
            
            # Calcula os quantis (limites do intervalo de credibilidade)
            intervals_matrix[i, 0] = np.quantile(param_chain, lower_bound)
            intervals_matrix[i, 1] = np.quantile(param_chain, upper_bound)
        
        return mean_vector, intervals_matrix

    def solucao(self):
        chain, chainflat = self.get_results()
        media, intervalo = self.mean_interval(chainflat)
        return media, intervalo

def plot_trace(chain, burn_in=300, labels=None):
    """
    Plota o trace dos walkers.
    
    Args:
        chain (numpy.ndarray): Cadeia no formato (walkers, passos, dimensões).
        burn_in (int): Número de passos a serem marcados como burn-in.
        labels (list, opcional): Lista com os nomes dos parâmetros.
    """
    ndim = chain.shape[-1]
    if labels is None:
        labels = [f"Param {i+1}" for i in range(ndim)]
        
    fig, axes = plt.subplots(ndim, 1, figsize=(10, 8), sharex=True)
    if ndim == 1:
        axes = [axes] # Garante que seja iterável para 1 dimensão

    for i in range(ndim):
        axes[i].plot(chain[:, :, i], color='black', alpha=0.3)
        axes[i].axvline(burn_in, color='red', linestyle='--', label='Burn in')
        axes[i].set_ylabel('Valor', fontsize=11)
        axes[i].set_title(f'Trace de {labels[i]}', fontsize=12)
        axes[i].grid(alpha=0.3)
        
    axes[ndim-1].set_xlabel('Passos', fontsize=12)    
    fig.tight_layout()
    fig.canvas.draw()
    return fig.canvas.buffer_rgba()

def plot_distribution(chainflat, x_real=None, titulo="Distribuição Posterior", labels=None):
    """
    Plota a distribuição posterior a partir da cadeia achatada.
    
    Args:
        chainflat (numpy.ndarray): Cadeia no formato achatado (passos, dimensões).
        x_real (list ou numpy.ndarray, opcional): Valores reais para plotar a linha de comparação.
        titulo (str): Título da figura.
        labels (list, opcional): Lista com os nomes dos parâmetros.
    """
    ndim = chainflat.shape[-1]
    if labels is None:
        labels = [f"Param {i+1}" for i in range(ndim)]
        
    fig, axes = plt.subplots(ndim, 1, figsize=(10, 8), sharex=True)
    if ndim == 1:
        axes = [axes] # Garante que seja iterável
        
    fig.suptitle(titulo, fontsize=16, y=0.96)
    
    for i in range(ndim):
        data = chainflat[:, i]    
        mean = np.mean(data)
        axes[i].hist(data, bins=60, color='gray', alpha=0.6, density=True)
        axes[i].axvline(mean, color='red', linestyle='--', linewidth=2, label='Média Posterior')
        
        if x_real is not None and len(x_real) > i:
            axes[i].axvline(x_real[i], color='blue', linestyle='--', linewidth=2, label='Real')
            
        axes[i].axvspan(mean-np.std(data), mean+np.std(data), alpha=0.2, color='gray', label='1 desvio padrão')
        axes[i].set_ylabel('Densidade', fontsize=11)
        axes[i].set_title(f'Posterior de {labels[i]}', fontsize=12)
        axes[i].legend(loc='upper right')
        axes[i].grid(alpha=0.3)

    axes[ndim-1].set_xlabel('Valor', fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(top=0.90) # Ajuste sutil para o suptitle não sobrepor o tracejado
    fig.canvas.draw()
    return fig
