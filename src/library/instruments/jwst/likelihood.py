import nifty8.re as jft


def model_wrap(model, target_domain=None):
    if target_domain is None:
        def wrapper(x):
            out = model(x)
            for x, val in x.items():
                out[x] = val
            return out
    else:
        def wrapper(x):
            out = model(x)
            for key in target_domain.keys():
                out[key] = x[key]
            return out
    return wrapper


def connect_likelihood_to_model(
    likelihood: jft.Likelihood,
    model: jft.Model
) -> jft.Likelihood:
    '''Connect the likelihood and model, this is necessery when some models are
    inside the likelihood.
    In this case the keys necessery are passed up the chain, such that white
    priors are passed to the respective keys of the likelihood.
    '''

    ldom = likelihood.domain.tree
    tdom = {t: ldom[t] for t in ldom.keys() if t not in model.target.keys()}
    mdom = tdom | model.domain

    model_wrapper = model_wrap(model, tdom)
    model = jft.Model(
        lambda x: jft.Vector(model_wrapper(x)),
        domain=jft.Vector(mdom)
    )

    return likelihood.amend(model, domain=model.domain)


def build_gaussian_likelihood(
    data,
    std
):
    if not isinstance(std, float):
        assert data.shape == std.shape

    var_inv = 1/(std**2)
    std_inv = 1/std

    return jft.Gaussian(
        data=data,
        noise_cov_inv=lambda x: x*var_inv,
        noise_std_inv=lambda x: x*std_inv,
    )
