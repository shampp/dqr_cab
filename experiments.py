from data import get_data, get_data_source
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
from pathlib import Path
from recommendation import *
from fasttxt import load_ft_model
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from plot import simple_plot, twinx_plot
from policy import policy_evaluation

def run_bandit_arms(dt, setting, bandit):
    import transformers
    transformers.logging.set_verbosity_error()
    n_rounds = 500
    candidate_ix = [2, 3, 5, 10]
    
    df, X, anchor_ids, noof_anchors = get_data(dt)
    ft = load_ft_model()
    src = get_data_source(dt)
    regret = {}
    avg_sim = {}
    avg_dst = {}
    model_dict = {}

    if (setting == 'scratch'):
        from transformers import BertTokenizer
        vocab = '../Data/semanticscholar/tokenizer/wordpiece/vocab.txt'
        tokenizer = BertTokenizer(vocab_file=vocab, unk_token='[unk]', cls_token='[bos]', sep_token='[sep]', bos_token='[bos]', eos_token='[eos]', pad_token='[pad]')
        sep = ' [sep] '
        special_tokens = ['[sep]', '[bos]']

        model_dest = '../Data/semanticscholar/model/gpt2/wordpiece'
        from transformers import GPT2LMHeadModel
        model_dict['GPT'] = {}
        model_dict['GPT']['model'] = GPT2LMHeadModel.from_pretrained(model_dest)
        model_dict['GPT']['tok'] = tokenizer
        model_dict['GPT']['spl_tok'] = special_tokens
        model_dict['GPT']['sep'] = sep

        model_dest = '../Data/semanticscholar/model/ctrl'
        from transformers import CTRLLMHeadModel
        model_dict['CTRL'] = {}
        model_dict['CTRL']['model'] = CTRLLMHeadModel.from_pretrained(model_dest)
        model_dict['CTRL']['tok'] = tokenizer
        model_dict['CTRL']['spl_tok'] = special_tokens
        model_dict['CTRL']['sep'] = sep
    else:
        from transformers import (TransfoXLLMHeadModel,TransfoXLTokenizer)
        model_dest = '../Data/semanticscholar/model/xl'
        model_dict['XL'] = {}
        model_dict['XL']['model'] = TransfoXLLMHeadModel.from_pretrained(model_dest)
        model_dict['XL']['tok'] = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
        model_dict['XL']['spl_tok'] = ['[<unk>]', '<unk>', '[bos]']
        model_dict['XL']['sep'] = ' [sep] '

        from transformers import (GPT2LMHeadModel,GPT2Tokenizer)
        model_dest = '../Data/semanticscholar/model/gpt2/pretrained'
        model_dict['GPT'] = {}
        model_dict['GPT']['model'] = GPT2LMHeadModel.from_pretrained(model_dest)
        model_dict['GPT']['tok'] = GPT2Tokenizer.from_pretrained('gpt2-medium')
        model_dict['GPT']['spl_tok'] = ['[sep]','<|endoftext|>']
        model_dict['GPT']['sep'] = ' [sep] '

        from transformers import (CTRLLMHeadModel, CTRLTokenizer)
        model_dest = '../Data/semanticscholar/model/ctrl/pretrained'
        model_dict['CTRL'] = {}
        model_dict['CTRL']['model'] = CTRLLMHeadModel.from_pretrained(model_dest)
        model_dict['CTRL']['tok'] = CTRLTokenizer.from_pretrained('sshleifer/tiny-ctrl')
        model_dict['CTRL']['spl_tok'] = ['[sep]']
        model_dict['CTRL']['sep'] = ' [sep] '
        
    for cand_sz in candidate_ix:
        regret[cand_sz] = {}
        avg_sim[cand_sz] = {}
        avg_dst[cand_sz] = {}
        log_file = Path('../Data/', src, 'logs', src+'_%s_%s_%d.log' %(setting,bandit,cand_sz))
        logging.basicConfig(filename = log_file, format='%(asctime)s : %(message)s', level=logging.INFO)
        logging.info("Running %s algorithm with candidate size %d" %(bandit, cand_sz))
        for anchor in anchor_ids:
            anchor_session_id = df.iloc[anchor]['session_id']
            seq_err, avg_sim[cand_sz][anchor], avg_dst[cand_sz][anchor] = policy_evaluation(bandit, setting, model_dict, X, true_ids, n_rounds, cand_sz, ft)
            regret[cand_sz][anchor] = regret_calculation(seq_err)
            #regret[cand_sz][anchor] = regret_calculation(policy_evaluation(bandit, setting, X, true_ids, n_rounds,cand_sz))
            logging.info("finished with regret calculation")
        logger = logging.getLogger()
        for hdlr in logger.handlers[:]:
            hdlr.close()
            logger.removeHandler(hdlr)

        
        simv = sum([sum(x)/noof_anchors for x in zip(*avg_sim[cand_sz].values())])/n_rounds
        dstv = sum([sum(x)/noof_anchors for x in zip(*avg_dst[cand_sz].values())])/n_rounds
        #print(sum(zip(*regret[bandit].values())))
        print("average similarity of %d is: %f" %(cand_sz, simv))
        print("average distance of %d is: %f" %(cand_sz, dstv))

    filename = 'arm_regret.pdf'
    simple_plot(regret,filename)


def run_bandit_round(dt,setting):
    from random import Random
    import transformers
    transformers.logging.set_verbosity_error()
    rnd = Random()
    rnd.seed(44)

    n_rounds = 500
    cand_set_sz = 3
    experiment_bandit = list() 
    df, X, anchor_ids, noof_anchors = get_data(dt)
    model_dict = {}
    if setting == 'pretrained':
        #experiment_bandit = ['EXP3', 'XL', 'GPT', 'CTRL']
        experiment_bandit = ['GPT', 'EXP3', 'EXP3-SS', 'DQR-CAB']

        from transformers import (TransfoXLLMHeadModel, TransfoXLTokenizer)
        model_dest = '../Data/semanticscholar/model/xl'
        model_dict['XL'] = {}
        model_dict['XL']['model'] = TransfoXLLMHeadModel.from_pretrained(model_dest)
        model_dict['XL']['tok'] = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
        model_dict['XL']['spl_tok'] = ['[<unk>]', '<unk>', '[bos]']
        model_dict['XL']['sep'] = ' [sep] '

        from transformers import (GPT2LMHeadModel, GPT2Tokenizer)
        model_dest = '../Data/semanticscholar/model/gpt2/pretrained'
        model_dict['GPT'] = {}
        model_dict['GPT']['model'] = GPT2LMHeadModel.from_pretrained(model_dest)
        model_dict['GPT']['tok'] = GPT2Tokenizer.from_pretrained('gpt2-medium')
        model_dict['GPT']['spl_tok'] = ['[sep]','<|endoftext|>']
        model_dict['GPT']['sep'] = ' [sep] '

        from transformers import (CTRLLMHeadModel, CTRLTokenizer)
        model_dest = '../Data/semanticscholar/model/ctrl/pretrained'
        model_dict['CTRL'] = {}
        model_dict['CTRL']['model'] = CTRLLMHeadModel.from_pretrained(model_dest)
        model_dict['CTRL']['tok'] = CTRLTokenizer.from_pretrained('sshleifer/tiny-ctrl')
        model_dict['CTRL']['spl_tok'] = ['[sep]']
        model_dict['CTRL']['sep'] = ' [sep] '

    ft = load_ft_model()
    regret = {}
    avg_sim = {}
    avg_dst = {}
    src = get_data_source(dt)

    for bandit in experiment_bandit:
        log_file = Path('../Data/', src, 'logs',src+'_%s_%s.log' %(setting, bandit))
        logging.basicConfig(filename = log_file, format='%(asctime)s : %(message)s', level=logging.INFO)
        logging.info("Running %s algorithm trained from %s" %(bandit,setting))
        regret[bandit] = {}
        avg_sim[bandit] = {}
        avg_dst[bandit] = {}

        for anchor in anchor_ids:
            anchor_session_id = df.iloc[anchor]['session_id']
            true_ids = df.index[df['session_id'] == anchor_session_id].tolist()
            true_ids.sort() #just in case if
            seq_err, avg_sim[bandit][anchor], avg_dst[bandit][anchor] = policy_evaluation(bandit, setting, model_dict, X, true_ids, n_rounds, cand_set_sz, ft)
            regret[bandit][anchor] = regret_calculation(seq_err)

        logger = logging.getLogger()
        for hdlr in logger.handlers[:]:
            hdlr.close()
            logger.removeHandler(hdlr)

        simv = sum([sum(x)/noof_anchors for x in zip(*avg_sim[bandit].values())])/n_rounds
        dstv = sum([sum(x)/noof_anchors for x in zip(*avg_dst[bandit].values())])/n_rounds
        #print(sum(zip(*regret[bandit].values())))
        print("average similarity of %s is: %f" %(bandit, simv))
        print("average distance of %s is: %f" %(bandit, dstv))

    filename = 'round_regret.pdf'
    simple_plot(regret,filename)

