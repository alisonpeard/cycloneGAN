import warnings
import numpy as np
import revdbayes

def cv_fn(data, u_vec, v_vec, n_u, n_v, use_rcpp, **cv_control):
    if callable(cv_control.get('prior')):
        use_rcpp = False
    if cv_control.get('prior') is None:
        cv_control['prior'] = "mdi"
    if cv_control.get('h_prior') is None or cv_control['h_prior'].get('min_xi') is None:
        cv_control.setdefault('h_prior', {})['min_xi'] = -1
    if isinstance(cv_control.get('prior'), str) and cv_control['prior'] == "mdi" and cv_control['h_prior'].get('a') is None:
        cv_control.setdefault('h_prior', {})['a'] = 0.6
    for_set_prior = {'prior': cv_control['prior'], 'model': "gp"}
    for_set_prior.update(cv_control['h_prior'])
    gp_prior = revdbayes.set_prior(**for_set_prior)
    cv_control.pop('prior', None)
    cv_control.pop('h_prior', None)
    if cv_control.get('bin_prior') is None:
        cv_control['bin_prior'] = "jeffreys"
    for_set_bin_prior = {'prior': cv_control['bin_prior']}
    for_set_bin_prior.update(cv_control.get('h_bin_prior', {}))
    bin_prior = revdbayes.set_bin_prior(**for_set_bin_prior)
    cv_control.pop('bin_prior', None)
    cv_control.pop('h_bin_prior', None)
    n = cv_control.get('n', 1000)
    cv_control.pop('n', None)
    in_rpost = any(name in revdbayes.rpost_rcpp.__code__.co_varnames for name in cv_control)
    in_ru = any(name in rust.ru_rcpp.__code__.co_varnames for name in cv_control)
    rogue_names = [name for name in cv_control if name not in revdbayes.rpost_rcpp.__code__.co_varnames and name not in rust.ru_rcpp.__code__.co_varnames]
    rogue_args = {name: cv_control[name] for name in rogue_names}
    if rogue_args:
        print("The following arguments have been ignored:")
        print(rogue_args)
        cv_control = {name: cv_control[name] for name in cv_control if name not in rogue_names}
    j_max = data.index(max(data))
    data_max = data[j_max]
    n_max = len(data_max)
    data_rm = data[:j_max] + data[j_max+1:]
    n_rm = len(data_rm)
    pred_perf = [[None] * n_v for _ in range(n_u)]
    pred_perf_rcpp = [[None] * n_v for _ in range(n_u)]
    gp_postsim = revdbayes.rpost_rcpp if use_rcpp else revdbayes.rpost
    for_post = {'n': n, 'model': "bingp", 'prior': gp_prior, 'bin_prior': bin_prior}
    for_post.update(cv_control)
    sim_vals = [[None] * 4 for _ in range(n * n_u)]
    for i in range(n_u):
        u = u_vec[i]
        try:
            temp = gp_postsim(data=data, thresh=u, **for_post)
        except:
            if for_post.get('trans') is None or for_post['trans'] == "none":
                for_post['trans'] = "BC"
            else:
                for_post['trans'] = "none"
            temp = gp_postsim(data=data, thresh=u, **for_post)
        try:
            temp_rm = gp_postsim(data=data_rm, thresh=u, **for_post)
        except:
            if for_post.get('trans') is None or for_post['trans'] == "none":
                for_post['trans'] = "BC"
            else:
                for_post['trans'] = "none"
            temp_rm = gp_postsim(data=data_rm, thresh=u, **for_post)
        theta = np.column_stack((temp['bin_sim_vals'], temp['sim_vals']))
        theta_rm = np.column_stack((temp_rm['bin_sim_vals'], temp_rm['sim_vals']))
        which_v = [j for j, v in enumerate(v_vec) if v >= u]
        v_vals = [v_vec[j] for j in which_v]
        pred_perf[i][which_v] = bloocv(z=data, theta=theta, theta_rm=theta_rm, u1=u, u2_vec=v_vals, z_max=data_max, z_rm=data_rm, n=n)
        which_rows = list(range(1 + (i - 1) * n, i * n + 1))
        sim_vals[which_rows] = np.column_stack((theta, np.full(n, i)))
    return {'pred_perf': pred_perf, 'u_vec': u_vec, 'v_vec': v_vec, 'sim_vals': sim_vals, 'n': n, 'for_post': for_post, 'use_rcpp': use_rcpp}