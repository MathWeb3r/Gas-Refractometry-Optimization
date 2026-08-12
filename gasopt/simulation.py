import numpy as np
from numpy.linalg import norm
from scipy import optimize as opt
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline

from gasopt import Gas, GasMix

# calcula o vetor b
def calculate_b(phi: np.ndarray, matriz: np.ndarray):
    return np.dot(matriz, phi)

# adiciona ruido a phi (com restrição de soma unitária = 1)
def generate_noise_phi(vector: np.ndarray, erro: float):
    sigma = erro * np.abs(vector)

    noise = np.random.normal(loc=0, scale=sigma, size=vector.shape)
    novo_vetor = vector + noise

    # Garante que não haja valores negativos
    novo_vetor = np.maximum(novo_vetor, 0) 
    
    # Garante a restrição física da soma unitária para frações molares
    return novo_vetor / np.sum(novo_vetor)

# adiciona ruido ao vetor de refratividades b (sem normalização unitária)
def generate_noise_b(vector: np.ndarray, erro: float):
    sigma = erro * np.abs(vector)
    noise = np.random.normal(loc=0, scale=sigma, size=vector.shape)
    return vector + noise


# calcula o vetor phi
# calculando o inverso
def caculate_phi(matrix: np.ndarray, refracVec: np.ndarray):
    return np.dot(np.linalg.inv(matrix), refracVec)

class SimulationRRM():
    """
    Simulate de inverse problem, with no constrains or bounds for the solution
    """
    def __init__(self, gases_id, lambdas: list, phi: list, erro: float, temp: float):
        self.gases = [Gas(id) for id in gases_id]
        self.mtx = GasMix(self.gases, lambdas, temp).get_matrix()
        self.lambas = np.array(lambdas) 
        self.phi = np.array(phi)
        self.erro = erro

    # simulando o sistema e problema inverso
    def _solve_inverse(self):

        # Refratividade da mistrua
        refrac = calculate_b(self.phi, self.mtx)    

        # add ruído
        refrac_noise = generate_noise_b(refrac, self.erro)

        # encontrando fração molar
        phi_simulado = caculate_phi(self.mtx, refrac_noise)

        return phi_simulado
    
    # simulando o sistema e problema inverso 
    # usando np.linalg.lstsq
    def _solve_inverse_lstsq(self):
        # Refratividade da mistrua
        refrac = calculate_b(self.phi, self.mtx)    

        # add ruído
        refrac_noise = generate_noise_b(refrac, self.erro)

        # encontrando fração molar
        phi_simulado, _, _, _ = np.linalg.lstsq(self.mtx, refrac_noise, rcond=None)

        return phi_simulado
    
    # norma quadrada para ser minimizada
    # usado no problema restrito
    def _sqrnorm(self, A, b, x):
        return np.linalg.norm((A @ x - b))
    
    def _gradient(self, A, b, x):
        return 2 * A.T @ (A @ x - b)

    def _solve_constrain(self):
        # melhorando a precisão numérica
        mtxA = self.mtx * 1e4

        # calcula b e já add ruido
        b_noisy = generate_noise_b(mtxA @ self.phi, erro=self.erro)

        # Restrição de igualdade: soma das frações molares deve ser igual a 1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
        
        # Restrições de caixa (limites): 0 <= x_i <= 1
        bounds = opt.Bounds(0.0, 1.0)
        
        # Chute inicial (solução da inversão direta)
        x0 = np.ones(len(b_noisy))/len(b_noisy)
        
        # Execução da otimização via SLSQP
        res = opt.minimize(
            fun=lambda x: self._sqrnorm(mtxA, b_noisy, x), 
            x0=x0, 
            jac=lambda x: self._gradient(mtxA, b_noisy, x), 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints)
        
        return res.x

    def run_simulation(self, n_iter: int = 1, method: str = 'normal'):
        sim_phi = np.zeros([n_iter, len(self.lambas)])

        if method == 'normal':
            simfunc = self._solve_inverse
        elif method == 'lstsq':
            simfunc = self._solve_inverse_lstsq
        elif method == 'constrain':
            simfunc = self._solve_constrain
        else:
            raise ValueError("Method must be in ['normal', 'lstsq']")

        for i in range(n_iter):
            sim_phi[i] = simfunc()

        return sim_phi

    def run_simulation_vec(self, n_iter: int = 1):
        sim_phi = np.zeros(n_iter)

        # o signature indica que o retorno do pyfunc
        # é uma array de tamanho 4 fixo
        vec = np.vectorize(
            pyfunc=lambda x: self._simulate_rrm(),
            signature='()->(4)')
        
        return vec(sim_phi)

class TikhonovSolverOld():
    def __init__(self, gases_id, lambdas: list, phi: list, erro: float, temp: float, rng_seed=0):
        self.gases = [Gas(id) for id in gases_id]
        self.mtx = GasMix(self.gases, lambdas, temp).get_matrix()
        self.lambas = np.array(lambdas) 
        self.phi = np.array(phi)
        self.erro = erro
        self.seed = rng_seed

        # dados do especificos
        self.alphas = np.logspace(-5, 0, 100)
        self.A = self.mtx*1e4

        if rng_seed != 0:
            rng = np.random.seed(rng_seed)

        self.set_keep_b(True)
        self.b_noisy = generate_noise(self.A @ self.phi, erro=self.erro)
    
    def set_keep_b(self, state):
        self.keep_b = bool(state)

    def _sqrnorm(self, A, b, alpha, x, x0):
        return np.sum((A @ x - b) ** 2) + (alpha ** 2) * np.sum((x-x0) ** 2)
    
    def _gradient(self, A, b, alpha, x, x0):
        return 2 * A.T @ (A @ x - b) + 2 * (alpha ** 2) * (x - x0)

    def minimize_constrain(self, A, b_noisy, alpha, x0):
        # Restrição de igualdade: soma das frações molares deve ser igual a 1
        constraints = {'type': 'eq', 
                       'fun': lambda x: np.sum(x) - 1.0}
        
        # Restrições de caixa (limites): 0 <= x_i <= 1
        bounds = opt.Bounds(0, 1.0)
        
        res = opt.minimize(
            fun=lambda x: self._sqrnorm(A, b_noisy, alpha, x, x0), 
            x0=x0, 
            jac=lambda x: self._gradient(A, b_noisy, alpha, x, x0), 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'ftol': 1e-14})
        

        return res.x

    def iter_alphas(self):
        # array pra receber os valores
        phis = np.zeros([len(self.alphas), len(self.b_noisy)])

        # Listas para armazenar as componentes da Curva L
        normas_residuo = []
        normas_solucao = []

        for i in range(len(phis)):
            phi_calc = self.minimize_constrain(
                A=self.A, 
                b_noisy=self.b_noisy, 
                alpha=self.alphas[i], 
                x0=np.ones(len(self.b_noisy))/len(self.b_noisy)
            )

            phis[i] = phi_calc

            # Cálculo das normas euclidianas quadradas para a Curva L
            res_norm = np.sum((self.A @ phi_calc - self.b_noisy) ** 2)
            sol_norm = np.sum(phi_calc ** 2)
            
            normas_residuo.append(res_norm)
            normas_solucao.append(sol_norm)

        return np.array(normas_residuo), np.array(normas_solucao), phis
    
    # metodo 1
    def curve_L(self, normas_residuo, normas_solucao):
        # Mapeamento exato das variáveis do PDF para o código:
        rho = normas_residuo              # Escala linear (rho)
        eta = normas_solucao              # Escala linear (eta)
        
        # Para calcular a derivada de rho em relação a alfa (rho'):
        # Garantimos que os dados passados para a Spline estão com alfa crescente
        idx_sort = np.argsort(self.alphas)
        alfa_sort = self.alphas[idx_sort]
        rho_sort = rho[idx_sort]
        
        # Criamos uma spline cúbica de rho em função de lambda para obter a derivada estável
        spline_rho = CubicSpline(alfa_sort, rho_sort)
        
        # Calculamos a derivada primeira rho' em todos os pontos de lambda originais
        rho_prime = spline_rho(self.alphas, nu=1)
        
        # --- Aplicação exata da EQUAÇÃO (6) do PDF ---
        # Numerador: η*ρ*[λ*η + λ²*ρ] + [η*ρ]² / ρ'
        termo_1 = eta * rho * (self.alphas * eta + (self.alphas**2) * rho)
        termo_2 = ((eta * rho)**2) / rho_prime
        numerador = termo_1 + termo_2
        
        # Denominador: (λ²*η² + ρ²)^(3/2)
        denominador = ((self.alphas**2) * (eta**2) + rho**2 )**(1.5)
        
        # Curvatura k(lambda)
        curvatura = numerador / denominador
        
        return curvatura

    # metodo 2
    def geometric_curvature(self, rho, eta):
        """
        Calcula a curvatura no espaço log-log utilizando splines suavizadas 
        para blindar o modelo contra ruído numérico do SLSQP.
        """
        # 1. Transformação para espaço log-log
        log_rho = np.log10(rho)
        log_eta = np.log10(eta)
        log_alphas = np.log10(self.alphas)
        
        # 2. Ordenação rigorosa (caso os alphas não estejam em ordem)
        idx = np.argsort(log_alphas)
        x_c = log_rho[idx]
        y_c = log_eta[idx]
        t = log_alphas[idx]
        
        # 3. Splines suavizadas (s > 0) paramétricas em função de log(alpha)
        # O fator de suavização 's' pode ser ajustado dependendo do ruído restante
        spline_x = UnivariateSpline(t, x_c, k=3, s=0.1)
        spline_y = UnivariateSpline(t, y_c, k=3, s=0.1)
        
        # 4. Derivadas analíticas das splines
        x_prime = spline_x.derivative(1)(t)
        x_double_prime = spline_x.derivative(2)(t)
        
        y_prime = spline_y.derivative(1)(t)
        y_double_prime = spline_y.derivative(2)(t)
        
        # 5. Cálculo da Curvatura Geométrica
        numerador = (x_prime * y_double_prime) - (y_prime * x_double_prime)
        denominador = (x_prime**2 + y_prime**2)**(1.5)
        
        curvatura = numerador / denominador
        
        # Mapeia de volta para a ordem original de alphas
        curvatura_orig_order = np.zeros_like(curvatura)
        curvatura_orig_order[idx] = curvatura

        # parece haver um offset no calculo
        # quando calculo a curvatura é de um ponto ao outro
        # e ela vale para o ponto do final
        #curvatura_orig_order = np.insert(curvatura_orig_order, 0, 0)
        #curvatura_orig_order = curvatura_orig_order[:-1]
        
        return curvatura_orig_order

    # metodo 3
    def curvature_svd(self):
        """
        Calcula a curvatura da curva L para uma sequência de alphas usando SVD.
        A: Matriz do problema
        b: Vetor de dados
        alpha: Escalar ou array de parâmetros de regularização (λ)
        """
        # 1. Calcula o SVD de A
        U, s, Vt = np.linalg.svd(self.A, full_matrices=False)
        beta = U.T @ self.b_noisy
        
        # no artigo esse calculo é pra forma em que alpha
        # está ao quadrado
        alphas = self.alphas

        # Garante que alpha seja um array para vetorização
        alphas = np.atleast_1d(alphas)
        curvatures = []
        
        for al in alphas:
            # Fatores de filtro de Tikhonov (fi)
            f = (s**2) / (s**2 + al**2)
            
            # Normas ao quadrado: eta (solução) e rho (resíduo)
            eta = np.sum((f * beta / s)**2)
            rho = np.sum(((1 - f) * beta)**2)
            
            # Componente b0 (se m > n, parte de b fora do espaço de colunas de A)
            # Se A for quadrada, b0_norm_sq será 0
            b0_norm_sq = np.sum(self.b_noisy**2) - np.sum(beta**2)
            rho += b0_norm_sq
            
            # Derivada de eta em relação a alpha (eta')
            eta_prime = -(4 / al) * np.sum((1 - f) * (f**2) * (beta**2) / (s**2))
            
            # Fórmula Hansen para a curvatura kappa em escala log-log
            numerator = 2 * eta * rho * (al**2 * eta_prime * rho + 2 * al * eta * rho + al**4 * eta * eta_prime)
            denominator = eta_prime * (al**2 * eta**2 + rho**2)**(1.5)
            
            kappa = numerator / denominator
            curvatures.append(kappa)
            
        return np.array(curvatures) if np.ndim(alphas) > 0 else curvatures[0]

    def solve(self, method=3):
        if self.keep_b:
            self.b_noisy = generate_noise(self.A @ self.phi, erro=self.erro)

        residual_norm, solution_norm, phis_result = self.iter_alphas()

        if method == 1:
            curvatura = self.curve_L(
                normas_residuo=residual_norm, 
                normas_solucao=solution_norm)
        elif method == 2:
            # curvatura da curva L
            curvatura = self.geometric_curvature(
                rho=residual_norm, 
                eta=solution_norm)
        elif method == 3:
            curvatura = self.curvature_svd()

        return TikSolution(
            res=residual_norm,
            sol=solution_norm,
            curv=curvatura,
            alphas=self.alphas
        )

class TikSolution():
    def __init__(self, res, sol, curv, alphas):
        self.residual_norm = res
        self.solution_norm = sol
        self.curvature = np.abs(curv)
        self.alphas = alphas

        # encontrando onde a curvatura é a maior
        optIdx = np.argmax(np.abs(self.curvature)) 
        self.opt_alpha = self.alphas[optIdx]
    
    def __str__(self):
        return f'Optimal Alpha: {self.opt_alpha}'

if __name__ == '__main__':
    from plotting import plot_curvaL_e_curvatura
    import matplotlib.pyplot as plt

    for i in range(1):
        tik = TikhonovSolver(
            gases_id= ['co2_1', 'ar_1', 'n2_1', 'o2'],
            lambdas=[0.74, 0.77, 0.82, 0.86],
            phi=[0.1, 0.2, 0.4, 0.3],
            erro=0.01,
            temp=273.15,
        )

        print('B com ruido:', tik.b_noisy)
        sol = tik.solve(method=3)

        plot_curvaL_e_curvatura(
            alfas=sol.alphas,
            normas_residuo=sol.residual_norm,
            normas_solucao=sol.solution_norm,
            curv=sol.curvature
        )

        print('curvatura', sol.curvature)
    plt.show()