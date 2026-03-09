import jax.numpy as jnp
import jax.scipy.special as jss
import jax.scipy.stats as jsst
import numpyro
import numpyro.distributions as dist


def truncated_normal_logpdf(x, mu, sigma, lower=0.0, upper=1.0):
    
    z = (x - mu) / sigma
    a = (lower - mu) / sigma
    b = (upper - mu) / sigma

    log_base = jsst.norm.logpdf(z) - jnp.log(sigma)
    log_norm = jnp.log(
        jss.ndtr(b) - jss.ndtr(a)
    )
    return log_base - log_norm


def samples_model(q_samples, lower=0.0, upper=1.0):
    
    q_samples = jnp.asarray(q_samples)

    mu = numpyro.sample("mu", dist.Uniform(0.0, 1.2))
    sigma = numpyro.sample("sigma", dist.Uniform(0.1, 0.2))
    
    logps = truncated_normal_logpdf(q_samples, mu, sigma, lower=lower, upper=upper)  # shape (Nobs, Nsamp)

    marginal_logls = jss.logsumexp(logps, axis=1)  # shape (Nobs,)

    log_neff = 2.0 * marginal_logls - jss.logsumexp(2.0 * logps, axis=1)
    neff = jnp.exp(log_neff)

    numpyro.deterministic("neff", neff)

    numpyro.factor("obs", jnp.sum(marginal_logls))