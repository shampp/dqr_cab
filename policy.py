from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from fasttxt import  get_query_embeddings

def policy_evaluation(bandit, setting, model_dict, X, true_ids, n_rounds, cand_set_sz, ft):
    if bandit == 'GPT':
        return run_gpt(setting, model_dict['GPT'], X, true_ids, n_rounds, cand_set_sz, ft)
    if bandit == 'EXP3':
        return run_exp3(setting, model_dict, X, true_ids, n_rounds, cand_set_sz, ft)
    if bandit == 'EXP3-SS':
        return run_exp3_ss(setting, model_dict, X, true_ids, n_rounds, cand_set_sz, ft)
    if bandit == 'DQR_CAB':
        return run_dqr_cab(setting, model_dict, X, true_ids, n_rounds, cand_set_sz, ft)

def regret_calculation(seq_error):
    t = len(seq_error)
    regret = [x / y for x, y in zip(seq_error, range(1, t + 1))]
    return regret

def get_distance(ft, curr_query, pred_query):
    pred = ' '.join(list(set(pred_query.split())))
    curr = ' '.join(list(set(curr_query.split())))
    p_vec = get_query_embeddings(ft, pred)
    c_vec = get_query_embeddings(ft, curr)
    return euclidean_distances(p_vec.reshape(1,-1),c_vec.reshape(1,-1))[0][0]

def get_similarity(ft, curr_query, pred_query):
    pred = ' '.join(list(set(pred_query.split())))
    curr = ' '.join(list(set(curr_query.split())))
    p_vec = get_query_embeddings(ft, pred)
    c_vec = get_query_embeddings(ft, curr)
    return cosine_similarity(p_vec.reshape(1,-1),c_vec.reshape(1,-1))[0][0]

def run_exp3(setting, model_dict, X, true_ids, n_rounds, cand_set_sz, ft):
    cand_set_sz = 25
    from random import Random
    rnd1 = Random()
    rnd1.seed(42)
    rnd2 = Random()
    rnd2.seed(99)
    random.seed(42)
    eta = 1e-3
    seq_error = np.zeros(shape=(n_rounds, 1))
    simv = np.zeros(shape=(n_rounds, 1))
    dstv = np.zeros(shape=(n_rounds, 1))
    r_t = 1
    w_t = dict()
    curr_id = true_ids[0]   #for curr_id in true_ids[:-1]:  #p_t = list()
    curr_query = X[curr_id]
    cand = get_recommendations(curr_query, cand_set_sz, model_dict, setting)
    cand_sz = len(cand)
    logger.info("candidate set are: {}".format(','.join(map(str, cand))))
    S = [0 for i in range(cand_sz)]
    
    for t in range(n_rounds):
        p = [ np.exp(eta*S[i]) for i in range(cand_sz) ]
        p = p/sum(p)
        curr_id = rnd1.choice(true_ids)   #for curr_id in true_ids[:-1]:  #p_t = list()
        curr_query = X[curr_id]
        logging.info("Running recommendations for id : %d" %(curr_id))
        logging.info("Corresponding query is : %s" %(curr_query))
        ground_actions = true_ids.copy()
        ground_actions.remove(curr_id)  #this is the possible set of actions that are correct
        ground_queries = X[ground_actions]
        cand.update(cand_t)
        ind = rnd2.choices(range(len(p_t)), weights=p_t)[0]
        logger.info("getting recommendation scores")
        score = get_recommendation_score(ground_queries,cand[ind])
        logger.info("recommendation score is: %f" %(score))
        if score >= 0.5:
            r_t = 1
            if (t > 0):
                seq_error[t] = seq_error[t-1]
        else:
            r_t = 0
            seq_error[t] = 1 if (t==0) else seq_error[t-1] + 1.0

        S[ind] = S[ind] +  1 - (1-r_t)/p_t[ind]

        simv[t] = get_similarity(ft, curr_query, w_k[ind])
        dstv[t] = get_distance(ft, curr_query, w_k[ind])

    return seq_error, simv , dstv


def run_exp3_ss(setting, model_dict, X, true_ids, n_rounds, cand_set_sz, ft):
    from random import Random
    rnd1 = Random()
    rnd1.seed(42)
    rnd2 = Random()
    rnd2.seed(99)
    random.seed(42)
    eta = 1e-3
    seq_error = np.zeros(shape=(n_rounds, 1))
    simv = np.zeros(shape=(n_rounds, 1))
    dstv = np.zeros(shape=(n_rounds, 1))
    r_t = 1
    w_t = dict()
    cand = set()

    for t in range(n_rounds):
        curr_id = rnd1.choice(true_ids)   #for curr_id in true_ids[:-1]:  #p_t = list()
        curr_query = X[curr_id]
        logging.info("Running recommendations for id : %d" %(curr_id))
        logging.info("Corresponding query is : %s" %(curr_query))
        ground_actions = true_ids.copy()
        ground_actions.remove(curr_id)  #this is the possible set of actions that are correct
        ground_queries = X[ground_actions]
        cand_t = get_recommendations(curr_query, cand_set_sz, model_dict, setting)
        tsz = len(cand)
        cand_sz = 1 if tsz == 0 else tsz
        cand_t = cand_t.difference(cand)
        tsz = len(cand_t)
        cand_t_sz = 1 if tsz == 0 else tsz
        if (t == 0):
            for q in cand_t:
                w_t[q] = eta/((1-eta)*cand_t_sz)
        else:
            W = sum([w_t[q] for q in cand])
            for q in cand_t:
                w_t[q] = (eta*W)/((1-eta)*cand_t_sz)

        w_k = list(w_t.keys())
        p_t = [ (1-eta)*w + eta/cand_sz for w in w_t.values() ]
        cand.update(cand_t)
        logger.info("candidate set are: {}".format(','.join(map(str, cand))))
        ind = rnd2.choices(range(len(p_t)), weights=p_t)[0]
        logger.info("getting recommendation scores")
        score = get_recommendation_score(ground_queries,w_k[ind])
        logger.info("recommendation score is: %f" %(score))
        if score >= 0.5:
            r_t = 1
            if (t > 0):
                seq_error[t] = seq_error[t-1]
        else:
            r_t = 0
            seq_error[t] = 1 if (t==0) else seq_error[t-1] + 1.0

        r_hat = r_t/p_t[ind]
        w_t[w_k[ind]] = w_t[w_k[ind]]*np.exp(eta*r_hat)

        simv[t] = get_similarity(ft, curr_query, w_k[ind])
        dstv[t] = get_distance(ft, curr_query, w_k[ind])
    return seq_error, simv , dstv


def run_dqr_cab(setting, model_dict, X, true_ids, n_rounds, cand_set_sz, ft):
    from random import Random
    rnd1 = Random()
    rnd1.seed(42)
    rnd2 = Random()
    rnd2.seed(99)
    random.seed(42)
    eta = 1e-3
    seq_error = np.zeros(shape=(n_rounds, 1))
    simv = np.zeros(shape=(n_rounds, 1))
    dstv = np.zeros(shape=(n_rounds, 1))
    r_t = 1
    w_t = dict()
    cand = set()
    s_t = 0

    for t in range(n_rounds):
        curr_id = rnd1.choice(true_ids)   #for curr_id in true_ids[:-1]:  #p_t = list()
        curr_query = X[curr_id]
        logging.info("Running recommendations for id : %d" %(curr_id))
        logging.info("Corresponding query is : %s" %(curr_query))
        ground_actions = true_ids.copy()
        ground_actions.remove(curr_id)  #this is the possible set of actions that are correct
        ground_queries = X[ground_actions]
        cand_t = get_recommendations(curr_query, cand_set_sz, model_dict, setting)
        tsz = len(cand)
        cand_sz = 1 if tsz == 0 else tsz
        cand_t = cand_t.difference(cand)
        tsz = len(cand_t)
        cand_t_sz = 1 if tsz == 0 else tsz
        if (t == 0):
            for q in cand_t:
                w_t[q] = eta/((1-eta)*cand_t_sz)
        else:
            W = sum([w_t[q] for q in cand])
            for q in cand_t:
                w_t[q] = (eta*W)/((1-eta)*cand_t_sz)

        w_k = list(w_t.keys())
        p_t = [ (1-eta)*w + eta/cand_sz for w in w_t.values() ]
        cand.update(cand_t)
        logger.info("candidate set are: {}".format(','.join(map(str, cand))))
        ind = rnd2.choices(range(len(p_t)), weights=p_t)[0]
        logger.info("getting recommendation scores")
        score = get_recommendation_score(ground_queries,w_k[ind])
        logger.info("recommendation score is: %f" %(score))
        if score >= 0.5:
            r_t = 1
            if (t > 0):
                seq_error[t] = seq_error[t-1]
        else:
            r_t = 0
            seq_error[t] = 1 if (t==0) else seq_error[t-1] + 1.0

        r_hat = r_t/p_t[ind]
        w_t[w_k[ind]] = w_t[w_k[ind]]*np.exp(eta*r_hat)

        simv[t] = get_similarity(ft, curr_query, w_k[ind])
        dstv[t] = get_distance(ft, curr_query, w_k[ind])
    return seq_error, simv , dstv


def run_gpt(setting, model_dict, X, true_ids, n_rounds, cand_set_sz, ft):
    from random import Random
    rnd = Random()
    rnd.seed(42)
    seq_error = np.zeros(shape=(n_rounds,1))
    simv = np.zeros(shape=(n_rounds, 1))
    dstv = np.zeros(shape=(n_rounds, 1))
    for t in range(n_rounds):
        curr_id = rnd.choice(true_ids)   #for curr_id in true_ids[:-1]:  #p_t = list()
        curr_query = X[curr_id]
        logging.info("Running recommendations for id : %d" %(curr_id))
        logging.info("Corresponding query is : %s" %(curr_query))
        ground_actions = true_ids.copy()
        ground_actions.remove(curr_id)  #this is the possible set of actions that are correct
        ground_queries = X[ground_actions]
        next_query = get_next_query(curr_query, model_dict, setting)
        score = get_recommendation_score(ground_queries, next_query)
        if score >= 0.5:
            if (t > 0):
                seq_error[t] = seq_error[t-1]
        else:
            seq_error[t] = 1 if (t==0) else seq_error[t-1] + 1.0

        simv[t] = get_similarity(ft, curr_query, next_query)
        dstv[t] = get_distance(ft, curr_query, next_query)

    return seq_error, simv, dstv    
