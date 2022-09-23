"""
Cython implementations of sampling from the semi-circle and related densities.
These routines are called many times within outer sampling loops and so it
is worth having compiled implementations. The gains come mainly from the
rejection sampling logic. The pure-numpy functions are included here for 
convenience only rather than for any performance gains.
The various magic contants below are presented and explained in 
https://arxiv.org/pdf/2203.08061.pdf.
"""
import numpy as np
from scipy import stats
from scipy.special import eval_hermite, factorial


cdef float SC_REJECTION_CONSTANT
cdef int DEGREES_FREEDOM
cdef float MIXTURE_PROB
SC_REJECTION_CONSTANT = 0.71 / 0.5
DEGREES_FREEDOM = 10
POWERS = [0, -1, -.5, -.25]
MIXTURE_PROBABLITY_COEFFS = np.array([ 0.10082775, -0.48568934,  0.64655707,  0.27249265])
REJECTION_BOUND_COEFFS = np.array([ 0.46252496,  1.05793943, -3.35210376,  3.30679049])


def sample_semicircle(int maxiters=1000):
    """
    Create a single sample from the semi-circle law with radius 2.
    Uses rejection sampling with a hand-crafted bimodal Gaussian
    mixture proposal distribution and emprirically deived rejection
    contant. This was found in practice to be more efficient than
    sampling from a Wigner matrix spectrum.
    
    :returns: A single sample from the semi-circle law.
    :rtype: float
    """ 
    cdef int iternum
    cdef float mean, proposal_point, proposal_density, ratio
    for iternum in range(maxiters):
        mean = -1 if np.random.randint(2)==0 else 1
        proposal_point = stats.norm.rvs(loc=mean, scale=.8)
        proposal_density = (stats.norm.pdf(proposal_point, loc=-1, scale=0.8) + stats.norm.pdf(proposal_point, loc=1, scale=0.8))/2
        ratio = semi_circle_density(proposal_point) / proposal_density
        if np.random.uniform() * SC_REJECTION_CONSTANT < ratio:
            return proposal_point


def semi_circle_density(float z):
    """
    Compute the semi-circle density with radius 2.

    :param float z: The point(s) at which to evaluate the density.

    :rtype: float
    """
    cdef float result
    if np.abs(z) < 2:
        result = np.sqrt(4 - z**2)/2/np.pi
    else:
        result = 0
    return result


def mixture_probability(n):
    return np.array([np.power(float(n), power) for power in POWERS]) @ MIXTURE_PROBABLITY_COEFFS


def semi_circle_student_mixture_proposal_density(float z, float mixture_prob):
    """
    Computes the density of a Student-t and semi-circle mixture.

    :param float,np.ndarray[float] z: The point(s) at which to evaluate the density.
    :param float mixture_prob: The probability of the Student-t component.

    :rtype: np.ndarray[float]
    """
    return semi_circle_density(z) * (1-mixture_prob) + stats.t.pdf(z, DEGREES_FREEDOM) * mixture_prob


def sample_semi_circle_student_mixture(float p):
    """
    Sample from a mixture of semi-circle random variable and a Student-t.

    :param float p: The probability of the Student-t component.
    """
    if np.random.binomial(1, p) == 1:
        return stats.t.rvs(DEGREES_FREEDOM)
    else:
        return sample_semicircle()


def finite_n_semi_circle_density(float x, int n):
    """
    The density defined by a truncated Hermite expansion of the semi-circle.

    :param float x: The value at which to evaluate the density.
    :param int n: The first n Hermite polynomials are included in the expansion.
    """
    prefactor = 1. / n
    hermite = sum(eval_hermite(ind, x / np.sqrt(2))**2 / (2**ind * factorial(ind)) for ind in range(n))
    return prefactor * hermite * stats.norm.pdf(x)


def finite_n_semi_circle_rejection_bound(int n):
    """
    A good rejection bound for the finite-n semi-circle distribution
    (i.e. the density defined by a truncated Hermite expansion of the semi-circle).
    """
    return 0.03 + np.array([np.power(float(n), power) for power in POWERS]) @ REJECTION_BOUND_COEFFS


def rejection_sample_finite_n_semi_circle(int n):
    """
    Sample from the finite-n truncated Hermite semi-circle expansion density.
    Rejection sampling is used with a semi-circle and Student-t mixture
    and hand-engineering good reject bounds.

    :param int n: The first n Hermite polynomials are included in the expansion.
    :rtype: float
    :returns: A sample from the density.
    """
    cdef int iternum
    cdef float proposal, proposal_density, u, bound, ratio

    bound  = finite_n_semi_circle_rejection_bound(n)

    for iternum in range(1000):
        proposal = sample_semi_circle_student_mixture(mixture_probability(n)) * np.sqrt(n)
        proposal_density = semi_circle_student_mixture_proposal_density(proposal/np.sqrt(n), mixture_probability(n)) / np.sqrt(n)
        ratio = finite_n_semi_circle_density(proposal, n) / proposal_density
        u = np.random.uniform()
        if u * bound < ratio:
            return proposal


def sample_finite_n_semi_circle(int n, int n_samples=1):
    """
    Wrapper around `rejection_sample_finite_n_semi_circle` to take multiple
    independent samples.
    """
    samples = np.zeros(n_samples)
    for ind in range(n_samples):
        samples[ind] = rejection_sample_finite_n_semi_circle(n)
    return samples
