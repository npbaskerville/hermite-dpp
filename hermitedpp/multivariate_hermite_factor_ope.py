from distutils.log import warn
import itertools as itt
from math import ceil
import warnings

from dppy.random_matrices import hermite_sampler_full
from dppy.utils import check_random_state, inner1d
import numpy as np
from scipy.special import eval_hermite, factorial
from scipy.stats import multivariate_normal

from semi_circle_cython_backend import sample_finite_n_semi_circle


class MultivariateHermiteFactorOPE:
    """
    Multivariate Hermite Orthogonal Polynomial Ensemble for Monte Carlo with Determinantal Point Processes.
    This corresponds to a continuous multivariate projection DPP with state space :math:`R^d` with respect to
    - reference measure :math:`\\mu(dx) = w(x) dx` where
        .. math::
            w(x) = e^{-x^2/2}.
    - kernel :math:`K` (see also :py:meth:`~dppy.multivariate_hermite_factor_ope.MultivariateHermiteFactorOPE.kernel`)
        .. math::
            K(x, y) = \\sum_{\\mathfrak{b}(k)=0}^{N-1}
                        P_{k}(x) P_{k}(y)
                    = \\Phi(x)^{\\top} \\Phi(y)
        where
        - :math:`k \\in \\mathbb{N}^d` is a multi-index ordered according to the ordering :math:`\\mathfrak{b}` (see :py:meth:`compute_ordering`)
        - :math:`P_{k}(x) = \\prod_{i=1}^d P_{k_i}(x_i)` is the product of orthonormal Hermite polynomials
            .. math::
                \\int_{-1}^{1}
                    P_{k}(u) P_{\\ell}(u)
                    w(u) d u
                = \\delta_{k\\ell}
            so that :math:`(P_{k})` are orthonormal w.r.t :math:`\\mu(dx)`
        - :math:`\\Phi(x) = \\left(P_{\\mathfrak{b}^{-1}(0)}(x), \\dots, P_{\\mathfrak{b}^{-1}(N-1)}(x) \\right)^{\\top}`
    """
    def __init__(self, n_points, dim):
        """
        :param int n_ponts: The number of points to sample.
        :param int dim: The dimension in which the process exists.

        .. note::
            The sampler employs a multivariate facorisation trick. As such the sampler is practically 
            limited to relatively low dimensions, for which `n_points` is at least roughly :math:`4^d + 1`.
        """
        self.single_factor_n = ceil(n_points ** (1 / dim))
        if self.single_factor_n < 5:
            warnings.warn(f"n_points ({n_points}) is so small, or dimension ({dim}) is so large,"
                          " that the factorisation sampler may be unreliable or very slow."
                          f"Conisder increase n_points to at least {5**dim}.")
        self.n_points = n_points
        self.dim = dim
        self.n_points_round_up = int(self.single_factor_n**self.dim)
        self.single_factor_n_delta = self.n_points_round_up - self.n_points

        self.nearest_factor_rejection_bound = (self.single_factor_n)**dim / self.n_points

        self.ordering = compute_ordering(self.n_points_round_up, self.dim)

        self.deg_max, self.degrees_1D_polynomials = compute_degrees_1D_polynomials(np.max(self.ordering, axis=0))

        self.norms_1D_polynomials = compute_norms_1D_polynomials(self.deg_max, self.dim)

        self.mass_of_mu = (2 * np.pi)**(self.dim / 2)

    def eval_w(self, x):
        """Evaluate the base measure density, i.e. standard Gaussian measure on :math:`R^d`."""
        return multivariate_normal(cov=np.eye(self.dim)).pdf(x) * (np.pi * 2)**(self.dim/2)

    def eval_multi_dimensional_polynomials(self, x, all=False):
        """Evaluate
        .. math::

            \\mathbf{\\Phi}(X)
                := \\begin{pmatrix}
                    \\Phi(x_1)^{\\top}\\\\
                    \\vdots\\\\
                    \\Phi(x_M)^{\\top}
                  \\end{pmatrix}
        :param np.ndarray x:
            :math:`M\\times d` array of :math:`M` points :math:`\\in \\mathbb{R}^d`.
        :param bool all: If to return all the computed multivariate polynomials or 
                         just the first n that we actually need.

        :return:
            :math:`\\mathbf{\\Phi}(X)` - :math:`M\\times N` array
        :rtype: array_like
        """
        scaled_x = np.atleast_2d(x)[:, None] / np.sqrt(2)
        poly_1D_hermite = eval_hermite(self.degrees_1D_polynomials, scaled_x) / self.norms_1D_polynomials
        if all:
            return np.prod(poly_1D_hermite[:, self.ordering, range(self.dim)], axis=2)
        else:
            return np.prod(poly_1D_hermite[:, self.ordering[:self.n_points], range(self.dim)], axis=2)


    def kernel(self, x, y=None, eval_pointwise=False):
        """Evalute :math:`\\left(K(x, y)\\right)_{x\\in X, y\\in Y}` if ``eval_pointwise=False`` or :math:`\\left(K(x, y)\\right)_{(x, y)\\in (X, Y)}` otherwise

        .. math::

            K(x, y) = \\sum_{\\mathfrak{b}(k)=0}^{N-1}
                        P_{k}(x) P_{k}(y)
                    = \\phi(x)^{\\top} \\phi(y)

        where

        - :math:`k \\in \\mathbb{N}^d` is a multi-index ordered according to the ordering :math:`\\mathfrak{b}`, :py:meth:`compute_ordering`

        - :math:`P_{k}(x) = \\prod_{i=1}^d P_{k_i}(x_i)` is the product of orthonormal Hermite polynomials on each dimension.

        :param array_like x: :math:`M\\times d` array of :math:`M` points :math:`\\in \\mathbb{R}^d`
        :param array_like,None y: :math:`M'\\times d` array of :math:`M'` points :math:`\\in \\mathbb{R}^d`
        :param bool eval_pointwise:
            sets pointwise evaluation of the kernel, if ``True``, :math:`x` and :math:`y` must have the same shape, see Returns

        :return:

            If ``eval_pointwise=False`` (default), evaluate the kernel matrix

            .. math::

                \\left(K(x, y)\\right)_{x\\in X, y\\in Y}

            If ``eval_pointwise=True`` kernel matrix
            Pointwise evaluation of :math:`K` as depicted in the following pseudo code output

            - if ``Y`` is ``None``

                - :math:`\\left(K(x, y)\\right)_{x\\in X, y\\in X}` if ``eval_pointwise=False``
                - :math:`\\left(K(x, x)\\right)_{x\\in X}` if ``eval_pointwise=True``

            - otherwise

                - :math:`\\left(K(x, y)\\right)_{x\\in X, y\\in Y}` if ``eval_pointwise=False``
                - :math:`\\left(K(x, y)\\right)_{(x, y)\\in (X, Y)}` if ``eval_pointwise=True`` (in this case x and y should have the same shape)
        :rtype: array_like
        """

        x = np.atleast_2d(x)

        if y is None or y is x:
            phi_x = self.eval_multi_dimensional_polynomials(x)
            if eval_pointwise:
                return inner1d(phi_x, phi_x, axis=1)
            else:
                return phi_x @ phi_x.T
        else:
            len_X = len(x)
            phi_xy = self.eval_multi_dimensional_polynomials(np.vstack((x, y)))
            if eval_pointwise:
                return inner1d(phi_xy[:len_X], phi_xy[len_X:], axis=1)
            else:
                return phi_xy[:len_X] @ phi_xy[len_X:].T

    def sample_chain_rule_proposal(self, rejection_trials_max=10000):
        """
        Use a rejection sampling mechanism to sample

        .. math::

            \\frac{1}{N} K(x, x) w(x) dx
            = \\frac{1}{N}
                \\sum_{\\mathfrak{b}(k)=0}^{N-1}
                \\left( \\frac{P_k(x)}{\\left\\| P_k \\right\\|} \\right)^2
                w(x)

        with proposal distribution given by the Student-t, semi-circle mixture.

        This is a two stage rejection sampler. We round up :math:`N` to :math:`n^d`,
        the next :math:`d`th power greater than :math:`N`. 

        This rounded up version of the kernel density is the proposal distribution.
        The bound in that case is simply :math:`n^d/N`.

        Sampling from the proposal distribution can be done using a factorisation
        trick, rewriting the density as a product over identical densities with only
        :math:`n` terms on each dimension. We can therefore sample each coordinate
        independently. 

        The univariate density converges quite quickly (in n) to the Wigner semi-circle.
        Therefore, the Wigner semi-circle can be utilised in the proposal distribution
        with good rejection bound. The semi-circle has compact support however,
        while the target density has sub-Gaussian tails. The student-t distribution 
        is mixed in to add some weight to the proposal tails and keep the rejection 
        bound valid. The various parameters, such as mixture probabilty and the 
        rejection bound were constructed empirically and are effective in practice.

        :param int rejection_trials_max: Maximum number of attempts to draw sample in rejection sampler.
        :rtype: float
        :returns: A sample from the chain rule proposal distribution.
        """
        for _ in range(rejection_trials_max):
            proposal_point = sample_finite_n_semi_circle(self.single_factor_n, n_samples=self.dim)
            phi_x = self.eval_multi_dimensional_polynomials(proposal_point, all=True)
            target_density = phi_x[:, :self.n_points] @ phi_x[:, :self.n_points].T
            proposal_density = phi_x @ phi_x.T
            uniform = np.random.uniform()
            if uniform * self.nearest_factor_rejection_bound < target_density / proposal_density:
                return proposal_point
        raise RuntimeError(f"Failure to sample after {rejection_trials_max} attempts.")

    def sample(self, nb_trials_max=10000, random_state=None, gue_1d=True):
        """Use the chain rule :cite:`HKPV06` (Algorithm 18) to sample :math:`\\left(x_{1}, \\dots, x_{N} \\right)` with density

        .. math::

            & \\frac{1}{N!}
                \\left(K(x_n,x_p)\\right)_{n,p=1}^{N}
                \\prod_{n=1}^{N} w(x_n)\\\\
            &= \\frac{1}{N} K(x_1,x_1) w(x_1)
            \\prod_{n=2}^{N}
                \\frac{
                    K(x_n,x_n)
                    - K(x_n,x_{1:n-1})
                    \\left[\\left(K(x_k,x_l)\\right)_{k,l=1}^{n-1}\\right]^{-1}
                    K(x_{1:n-1},x_n)
                    }{N-(n-1)}
                    w(x_n)\\\\
            &= \\frac{\\| \\Phi(x) \\|^2}{N} \\omega(x_1) d x_1
            \\prod_{n=2}^{N}
                \\frac{\\operatorname{distance}^2(\\Phi(x_n), \\operatorname{span}\\{\\Phi(x_p)\\}_{p=1}^{n-1})}
                {N-(n-1)}
            \\omega(x_n) d x_n

        The order in which the points were sampled can be forgotten to obtain a valid sample of the corresponding DPP

        - :math:`x_1 \\sim \\frac{1}{N} K(x,x) w(x)` using :py:meth:`sample_chain_rule_proposal`

        - :math:`x_n | Y = \\left\\{ x_{1}, \\dots, x_{n-1} \\right\\}`, is sampled using rejection sampling with proposal density :math:`\\frac{1}{N} K(x,x) w(x)` and rejection bound \\frac{N}{N-(n-1)}

            .. math::

                \\frac{1}{N-(n-1)} [K(x,x) - K(x, Y) K_Y^{-1} K(Y, x)] w(x)
                \\leq \\frac{N}{N-(n-1)} \\frac{1}{N} K(x,x) w(x)

        .. note::

            Using the gram structure :math:`K(x, y) = \\Phi(x)^{\\top} \\Phi(y)` the numerator of the successive conditionals reads

            .. math::

                K(x, x) - K(x, Y) K(Y, Y)^{-1} K(Y, x)
                &= \\operatorname{distance}^2(\\Phi(x_n), \\operatorname{span}\\{\\Phi(x_p)\\}_{p=1}^{n-1})\\\\
                &= \\left\\| (I - \\Pi_{\\operatorname{span}\\{\\Phi(x_p)\\}_{p=1}^{n-1}} \\phi(x)\\right\\|^2

            which can be computed simply in a vectorized way.
            The overall procedure is akin to a sequential Gram-Schmidt orthogonalization of :math:`\\Phi(x_{1}), \\dots, \\Phi(x_{N})`.

        :param int nb_trials_max: The maximum number of rejection sampling attempts.
        :param bool gue_1d: Should we use the GUE spectral formulation to sample 1d processes.

        :returns: Samples from the process. Shape n_samples x dimesion.
        :rtype: np.ndarray
        """
        rng = check_random_state(random_state)
        if self.dim == 1 and gue_1d:
            sample = hermite_sampler_full(self.n_points, random_state=rng)[:, None]
            return sample

        sample = np.zeros((self.n_points, self.dim))
        phi = np.zeros((self.n_points, self.n_points))

        for n in range(self.n_points):
            for trial in range(nb_trials_max):
                sample[n] = self.sample_chain_rule_proposal()
                phi[n] = self.eval_multi_dimensional_polynomials(sample[n])
                K_xx = phi[n].dot(phi[n])
                phi[n] -= phi[:n].dot(phi[n]).dot(phi[:n])
                schur = phi[n].dot(phi[n])

                ratio = schur / K_xx
                if rng.rand() < ratio:
                    phi[n] /= np.sqrt(schur)
                    break
            else:
                print('conditional x_{} | x_1,...,x_{}, rejection fails after {} proposals'.format(n + 1, n, trial))
        return sample


def compute_ordering(N, d):
    """
    Compute the ordering of the multi-indices :math:`\\in\\mathbb{N}^d` defining
    the order between the multivariate monomials as described in Section 2.1.3 of :cite:`BaHa16`.

    For instance, for :math:`N=12, d=2`

    .. code:: python

        [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2), (0, 3), (1, 3), (2, 3)]

    .. seealso::

        - :cite:`BaHa16` Section 2.1.3

    :param int N: Number of polynomials :math:`(P_k)` considered to build the kernel 
                  :py:meth:`~dppy.multivariate_jacobi_ope.MultivariateHermiteOPE.K` 
                  (number of points of the corresponding :py:class:`MultivariateHermiteOPE`)

    :param int d: Size of the multi-indices :math:`k\\in \\mathbb{N}^d` characterizing the
                    _degree_ of :math:`P_k` (ambient dimension of the points x_{1}, \\dots, x_{N} \\in [-1, 1]^d)

    :returns: Array of size :math:`N\\times d` containing the first :math:`N` multi-indices 
                :math:`\\in\\mathbb{N}^d` in the order prescribed by the ordering 
                :math:`\\mathfrak{b}` :cite:`BaHa16` Section 2.1.3
    :rtype: np.ndarray
    """
    layer_max = np.floor(N**(1.0 / d)).astype(np.int16)
    orders = itt.product(range(layer_max + 1), repeat=d)
    return sorted(orders, key=lambda inds: max(inds))[:N]


def compute_norms_1D_polynomials(deg_max, dim):
    """
    Compute the norms of the one dimensional Hermite polynomials.

    .. note::
        These numbers are big! We must stay in 64-bit precision to avoid
        overflows for even quite small values of deg_max.

    :param int deg_max: The maximum Hermite polynomial degree to compute.
    :param int dim: The number of compies in the dimension axis.
    """
    inds = np.arange(deg_max + 1).astype(np.float64)
    return np.tile(np.sqrt(factorial(inds) * 2**inds), (dim, 1)).T * (2 * np.pi)**0.25


def compute_degrees_1D_polynomials(max_degrees):
    """
    deg[i, j] = i if i <= max_degrees[j] else 0
    """
    max_deg, dim = max(max_degrees), len(max_degrees)
    degrees = np.tile(np.arange(max_deg + 1)[:, None], (1, dim))
    degrees[degrees > max_degrees] = 0

    return max_deg, degrees
