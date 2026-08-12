from typing import Optional, Union, List
import numpy as np
from scipy import optimize as opt

class ConstrainSolver:
    """
    Resolvedor de problemas inversos de refratometria de gases via Otimização Restrita (SLSQP).

    Esta classe formula e resolve a minimização da norma quadrada do resíduo ||A*x - b||^2
    sujeita a restrições físicas:
    - A soma das frações molares deve ser igual a 1 (sum(x) = 1)
    - Cada fração molar deve estar no intervalo [0, 1] (0 <= x_i <= 1)
    """

    def __init__(
        self,
        mtx: np.ndarray,
        sigma_mtx: np.ndarray,
        b_obs: np.ndarray,
        sigma_b: np.ndarray,
        x0: Optional[np.ndarray] = None,
        init_strategy: str = 'uniform',
        verbose: bool = False
    ):
        """
        Inicializa o resolvedor com as matrizes de refratividade, observações e estratégia de chute inicial.

        Parâmetros
        ----------
        mtx : np.ndarray
            Matriz de refratividade dos gases componentes (A), dimensão (M, N).
        sigma_mtx : np.ndarray
            Matriz de incertezas/erros dos elementos de mtx.
        b_obs : np.ndarray
            Vetor de refratividades observadas/medidas (b), dimensão (M,).
        sigma_b : np.ndarray
            Vetor de incertezas/erros associados a b_obs.
        x0 : Optional[np.ndarray], opcional
            Chute inicial customizado para as frações molares x, dimensão (N,).
            Se fornecido, substitui a estratégia de inicialização automática.
        init_strategy : str, padrão 'lstsq'
            Estratégia de determinação do chute inicial x0 quando x0 não for fornecido:
            - 'lstsq' / 'warm_start': Mínimos Quadrados / NNLS projetados no simplex (recomendado).
            - 'uniform': Centroide uniforme [1/N, ..., 1/N]^T.
            - 'multi_start': Testa múltiplos pontos de partida e escolhe o de menor resíduo.
        """
        # Melhora a precisão numérica multiplicando matrizes e observações por 1e4
        self.mtx = mtx * 1e4
        self.b_obs = b_obs * 1e4
        self.sigma_mtx = sigma_mtx * 1e4
        self.sigma_b = sigma_b * 1e4

        self.n_gases = self.mtx.shape[1]
        self.init_strategy = init_strategy.lower()

        if x0 is not None:
            self.x0 = np.asarray(x0, dtype=float)
        else:
            self.x0 = self._compute_initial_guess(self.init_strategy)

        self.verbose = verbose

    @staticmethod
    def _project_to_simplex(v: np.ndarray) -> np.ndarray:
        r"""
        Projeta um vetor v no simplex de probabilidade \Delta^N (sum(x) = 1, x_i >= 0)
        utilizando projeção Euclidiana exata (Duchi et al., 2008).
        """
        n = len(v)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho_idx = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1.0))[0]
        if len(rho_idx) == 0:
            return np.ones(n) / n
        rho = rho_idx[-1]
        theta = (cssv[rho] - 1.0) / (rho + 1)
        w = np.maximum(v - theta, 0.0)
        sum_w = np.sum(w)
        if sum_w > 0:
            return w / sum_w
        if self.verbose:
            print("Falhou.")
        return np.ones(n) / n

    def _compute_initial_guess(self, strategy: str) -> np.ndarray:
        """
        Calcula o vetor de chute inicial x0 baseado na estratégia solicitada.
        """
        if strategy == 'uniform':
            return np.ones(self.n_gases) / self.n_gases

        elif strategy in ('lstsq', 'warm_start'):
            try:
                # 1. Tenta mínimos quadrados não-negativos (NNLS)
                if self.verbose:
                    print("Tentando NNLS...")
                x_nnls, _ = opt.nnls(self.mtx, self.b_obs)
                sum_nnls = np.sum(x_nnls)
                if sum_nnls > 0:
                    return x_nnls / sum_nnls
            except Exception:
                if self.verbose:
                    print("Falhou.")

            # 2. Fallback: solução via mínimos quadrados regularizada / pseudo-inversa truncada
            try:
                if self.verbose:
                    print('Tentando o fallback para LSTSQ...')
                x_lstsq, _, _, _ = np.linalg.lstsq(self.mtx, self.b_obs, rcond=None)
                return self._project_to_simplex(x_lstsq)
            except Exception:
                if self.verbose:
                    print("Falhou.")
                return np.ones(self.n_gases) / self.n_gases

        elif strategy == 'multi_start':
            # Para multi_start no __init__, usa lstsq como chute base inicial
            return self._compute_initial_guess('lstsq')

        else:
            return np.ones(self.n_gases) / self.n_gases

    def _sqrnorm(self, A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
        """
        Calcula o valor da função objetivo (norma quadrada do resíduo ||A*x - b||^2).
        """
        return float(np.sum((A @ x - b) ** 2))

    def _gradient(self, A: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Calcula o gradiente analítico da função objetivo: 2 * A.T @ (A @ x - b).
        """
        return 2 * A.T @ (A @ x - b)

    def _solve_single(self, x0: np.ndarray) -> opt.OptimizeResult:
        """
        Executa uma única otimização SLSQP a partir de um chute inicial x0 dado.
        """
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
        bounds = opt.Bounds(0.0, 1.0)
        res = opt.minimize(
            fun=lambda x: self._sqrnorm(self.mtx, self.b_obs, x),
            x0=x0,
            jac=lambda x: self._gradient(self.mtx, self.b_obs, x),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-14}
        )
        return res

    def solve(self, init_strategy: Optional[str] = None) -> np.ndarray:
        """
        Resolve a otimização restrita utilizando o algoritmo SLSQP.

        Restrições aplicadas:
        - Igualdade: sum(x) = 1.0 (soma das frações molares)
        - Limites de caixa: 0.0 <= x_i <= 1.0

        Parâmetros
        ----------
        init_strategy : Optional[str]
            Sobrescreve a estratégia de inicialização para esta execução.

        Retorna
        -------
        np.ndarray
            Vetor de frações molares otimizadas x.
        """
        strategy = (init_strategy or self.init_strategy).lower()

        if strategy == 'multi_start':
            # Gerar múltiplos candidatos a chute inicial no simplex
            candidates = [
                self._compute_initial_guess('lstsq'),
                self._compute_initial_guess('uniform')
            ]
            # Adiciona os vértices do simplex (frações puras de cada gás)
            for i in range(self.n_gases):
                e_i = np.zeros(self.n_gases)
                e_i[i] = 1.0
                candidates.append(e_i)

            # Adiciona amostras aleatórias via distribuição de Dirichlet
            rng = np.random.default_rng(42)
            for _ in range(5):
                candidates.append(rng.dirichlet(np.ones(self.n_gases)))

            best_res = None
            best_fun = np.inf

            for x_cand in candidates:
                res = self._solve_single(x_cand)
                # Verifica conformidade de restrições
                violation = abs(np.sum(res.x) - 1.0)
                if violation < 1e-3 and res.fun < best_fun:
                    best_fun = res.fun
                    best_res = res

            if best_res is None:
                best_res = self._solve_single(self.x0)

            x_opt = best_res.x
        else:
            x0 = self.x0 if init_strategy is None else self._compute_initial_guess(strategy)
            res = self._solve_single(x0)
            x_opt = res.x

        # Garante que a solução final esteja rigorosamente no simplex
        return self._project_to_simplex(x_opt)

    solve_constrain = solve
