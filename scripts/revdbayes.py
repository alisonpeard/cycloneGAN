import warnings
import numpy as np


def get_prior(prior=["norm", "loglognorm", "mdi", "flat", "flatflat", "jeffreys", "beta", "prob", "quant"], model=["gev", "gp", "pp", "os"], **kwargs):
    if len(model) > 1:
        print("model not supplied: model == \"gev\" has been used.", immediate=True)
    model = match_arg(model)
    if callable(prior):
        temp = {"prior": prior, **kwargs}
        temp["trendsd"] = 0
        def new_prior(pars, **kwargs):
            return prior(pars, **kwargs)
        temp["prior"] = new_prior
        return {"temp": temp, "class": "evprior", "model": model}
    if isinstance(prior, "externalptr"):
        temp = {"prior": prior, **kwargs}
        return {"temp": temp, "class": "evprior", "model": model}
    prior = match_arg(prior)
    temp = {
        "gp": gp_prior(prior, **kwargs),
        "gev": gev_prior(prior, **kwargs),
        "pp": gev_prior(prior, **kwargs),
        "os": gev_prior(prior, **kwargs)
    }
    return {"temp": temp, "class": "evprior", "model": model}


def gp_prior(prior = ["norm", "mdi", "flat", "flatflat", "jeffreys", "beta"], **kwargs):
    prior = match_arg(prior)
    temp = {"prior": "gp_" + prior}
    temp.update(kwargs)
    
    if prior == "jeffreys" and temp.get("min_xi") is None:
        temp["min_xi"] = -1/2
    
    if prior == "mdi" and temp.get("min_xi") is None:
        temp["min_xi"] = -1
    
    if prior == "beta" and temp.get("min_xi") is None:
        temp["min_xi"] = -1/2
    
    if prior == "beta" and temp.get("max_xi") is None:
        temp["max_xi"] = 1/2
    
    temp["min_xi"] = max(temp.get("min_xi", float("-inf")), float("-inf"))
    temp["max_xi"] = min(temp.get("max_xi", float("inf")), float("inf"))
    
    if prior == "mdi" and temp.get("a") is None:
        temp["a"] = 1
    
    if prior == "beta" and temp.get("pq") is None:
        temp["pq"] = [6, 9]
    
    hpar_vec = {
        "norm": ["mean", "cov"],
        "mdi": "a",
        "flat": None,
        "jeffreys": None,
        "beta": "pq"
    }[prior]
    
    hpar_vec += ["min_xi", "max_xi", "upper"]
    
    temp = hpar_drop(temp, hpar_vec)
    
    if temp.get("min_xi") is not None and temp.get("max_xi") is not None:
        if temp["min_xi"] >= temp["max_xi"]:
            raise ValueError("min_xi must be less than max_xi")
    
    if temp.get("min_xi") is not None:
        if prior == "mdi" and np.isinf(temp["min_xi"]):
            raise ValueError("If min_xi=-Inf then the MDI posterior is not proper: min_xi must be finite.")
        
        if prior == "jeffreys" and temp["min_xi"] < -1/2:
            temp["min_xi"] = -1/2
            warnings.warn("min_xi < -1/2 does not make sense for the Jeffreys' prior.")



def gev_prior(prior=["norm", "loglognorm", "mdi", "flat", "flatflat", "beta", "prob", "quant"], **kwargs):
    prior = match_arg(prior)
    temp = {"prior": "gev_" + prior}
    temp.update(kwargs)
    if prior == "mdi" and temp.get("min_xi") is None:
        temp["min_xi"] = -1
    if prior == "beta" and temp.get("min_xi") is None:
        temp["min_xi"] = -1/2
    if prior == "beta" and temp.get("max_xi") is None:
        temp["max_xi"] = 1/2
    temp["min_xi"] = max(temp.get("min_xi", float("-inf")), float("-inf"))
    temp["max_xi"] = min(temp.get("max_xi", float("inf")), float("inf"))
    if prior == "mdi" and temp.get("a") is None:
        temp["a"] = 0.577215664901532
    if prior == "beta" and temp.get("pq") is None:
        temp["pq"] = [6, 9]
    hpar_vec = {
        "norm": ["mean", "cov"],
        "loglognorm": ["mean", "cov"],
        "mdi": "a",
        "flat": None,
        "beta": "pq",
        "prob": ["quant", "alpha"],
        "quant": ["prob", "shape", "scale"]
    }[prior]
    hpar_vec += ["min_xi", "max_xi"]
    temp = hpar_drop(temp, hpar_vec)
    if temp.get("min_xi") is not None and temp.get("max_xi") is not None:
        if temp["min_xi"] >= temp["max_xi"]:
            raise ValueError("min_xi must be less than max_xi")
    if temp.get("min_xi") is not None:
        if prior == "mdi" and np.isinf(temp["min_xi"]):
            raise ValueError("If min_xi=-Inf then the MDI posterior is not proper: min_xi must be finite.")
    if prior == "norm" or prior == "loglognorm":
        mean = temp["mean"]
        cov = temp["cov"]
        if len(mean) != 3 or type(mean) != list:
            raise ValueError("mean must be a list of length 3")


def match_arg(arg, choices, several_ok=False):
    if choices is None:
        formal_args = sys._getframe(1).f_code.co_varnames
        choices = eval(formal_args[formal_args.index(arg)], sys._getframe(1).f_globals, sys._getframe(1).f_locals)
    if arg is None:
        return choices[0]
    elif not isinstance(arg, str):
        raise ValueError("'arg' must be None or a string")
    if not several_ok:
        if arg == choices:
            return arg[0]
        if len(arg) > 1:
            raise ValueError("'arg' must be of length 1")
    elif len(arg) == 0:
        raise ValueError("'arg' must be of length >= 1")
    i = [choices.index(x) + 1 if x in choices else 0 for x in arg]
    if all(x == 0 for x in i):
        raise ValueError(f"'arg' should be one of {', '.join(choices)}")
    i = [x for x in i if x > 0]
    if not several_ok and len(i) > 1:
        raise ValueError("there is more than one match in 'match_arg'")
    return [choices[x - 1] for x in i]


def hpar_drop(x_list, hpar_vec):
    to_drop = [i+1 for i, name in enumerate(x_list[1:]) if name not in hpar_vec]
    
    if len(to_drop) == 1:
        print("This user-supplied argument is unused and has been dropped:", x_list[to_drop[0]])
    elif len(to_drop) > 1:
        print("The following user-supplied arguments are unused and have been dropped:")
        for i in to_drop:
            print(x_list[i])
    
    x_list = [x_list[i] for i in range(len(x_list)) if i not in to_drop]
    return x_list