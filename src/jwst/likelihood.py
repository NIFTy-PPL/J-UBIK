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

    ldom = likelihood.domain.tree
    tdom = {t: ldom[t] for t in ldom.keys() if t not in model.target.keys()}
    mdom = {t: model.domain[t] for t in model.domain.keys()}
    mdom.update(tdom)

    model_wrapper = model_wrap(model, tdom)

    model = jft.Model(
        lambda x: jft.Vector(model_wrapper(x)),
        domain=jft.Vector(mdom)
    )

    return likelihood.amend(model, domain=model.domain)
