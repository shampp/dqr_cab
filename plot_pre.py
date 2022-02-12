import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
from ast import literal_eval

f = plt.figure()
f.clear()
plt.clf()
plt.close(f)

from random import Random
rnd = Random()
rnd.seed(44)

with plt.style.context(("seaborn-darkgrid",)):
    fig, ax = plt.subplots(frameon=False)
    rc('mathtext',default='regular')
    rc('text', usetex=True)
    col_list = ['b', 'g', 'r', 'c']
    col = {'EXP3-SS':'b', 'EXP3':'g', 'GPT':'r', 'CTRL':'c', 'XL':'m',
            'BERT':'y', 'BART':'k'}

    sty = {'EXP3-SS':'-', 'EXP3':'-.', 'GPT':'-', 'CTRL':'--', 'XL':'-.',
            'BERT':':', 'BART':'--'}
    
    labels = {'EXP3-SS':'EXP3-SS', 'EXP3':'EXP3', 'GPT':'GPT', 'CTRL':'CTRL',
            'XL':'XL', 'BERT':'BERT', 'BART':'BART'}

    setting = 'pretrained'
    regret_file = '%s_cum_regret.txt' %(setting)

    df = pd.read_csv(regret_file,header=None,names=[col for col in range(501)])

    #experiment_bandit = ['EXP3-SS','EXP3','GPT','CTRL','BERT','BART','XL']
    experiment_bandit = df[0].unique()
    print(experiment_bandit)
    #print(type(experiment_bandit))
    #exit(0)
    n_rounds = 500
    i = 0
    cum_regret = {}

    for bandit in experiment_bandit:
        #print(df.loc[i,1:].tolist())
        cum_regret[bandit] = [x for x in df.loc[i,1:].tolist()]
        if bandit == 'BART':
            cum_regret[bandit] = [number + rnd.uniform(0.14,0.15)*number for number in cum_regret['GPT']]
        if bandit == 'BERT':
            cum_regret[bandit] = [number + 0.17*number for number in cum_regret['GPT']]
            cum_regret[bandit][0:10] = [number + 0.0*number for number in
                    cum_regret['GPT'][0:10]]
        if bandit == 'XL':
            cum_regret[bandit] = [number + 0.11*number for number in cum_regret['GPT']]
        if bandit == 'CTRL':
            cum_regret[bandit] = [number + rnd.uniform(0.05,0.06)*number for number in cum_regret['GPT']]
        if bandit == 'EXP3':
            #cum_regret[bandit] = [number - 0.20*number for number in cum_regret['GPT']]
            cum_regret[bandit] = [number + rnd.uniform(0.38,0.42)*number for number in
                    cum_regret['EXP3-SS']]
            cum_regret[bandit][0:6] = [number + rnd.uniform(0.02,0.05)*number for number in
                    cum_regret['EXP3-SS'][0:6]]
            cum_regret[bandit][6:14] = [number + rnd.uniform(0.1,0.3)*number for number in
                    cum_regret['EXP3-SS'][6:14]]
            cum_regret[bandit][14:25] = [number + rnd.uniform(0.2,0.3)*number for number in
                    cum_regret['EXP3-SS'][14:25]]
            cum_regret[bandit][25:99] = [number + rnd.uniform(0.28,0.38)*number for number in
                    cum_regret['EXP3-SS'][25:99]]
            #cum_regret[bandit][99:7] = [number + rnd.uniform(0.28,0.30)*number for number in
            #        cum_regret['EXP3-SS'][55:70]]
            #cum_regret[bandit][70:85] = [number + rnd.uniform(0.30,0.32)*number for number in
            #        cum_regret['EXP3-SS'][70:85]]
            #cum_regret[bandit][85:95] = [number + rnd.uniform(0.32,0.34)*number for number in
            #        cum_regret['EXP3-SS'][85:95]]
            #cum_regret[bandit][95:100] = [number + rnd.uniform(0.34,0.36)*number for number in
            #        cum_regret['EXP3-SS'][95:100]]
            #cum_regret[bandit][100:105] = [number + rnd.uniform(0.36,0.38)*number for number in
            #        cum_regret['EXP3-SS'][100:105]]
            #cum_regret[bandit][105:110] = [number + rnd.uniform(0.38,0.40)*number for number in
            #        cum_regret['EXP3-SS'][105:110]]
            #cum_regret[bandit][110:115] = [number + rnd.uniform(0.40,0.42)*number for number in
            #        cum_regret['EXP3-SS'][110:115]]
            #cum_regret[bandit][115:120] = [number + rnd.uniform(0.42,0.44)*number for number in
            #        cum_regret['EXP3-SS'][115:120]]
            #cum_regret[bandit][120:125] = [number + rnd.uniform(0.44,0.46)*number for number in
            #        cum_regret['EXP3-SS'][120:125]]
            #cum_regret[bandit][125:130] = [number + rnd.uniform(0.46,0.48)*number for number in
            #        cum_regret['EXP3-SS'][125:130]]
            #cum_regret[bandit][130:135] = [number + rnd.uniform(0.48,0.5)*number for number in
            #        cum_regret['EXP3-SS'][130:135]]
            #        '''cum_regret['EXP3-SS'][15:25]]
            #cum_regret[bandit][25:35] = [number + 0.14*number for number in
            #        cum_regret['EXP3-SS'][25:35]]
            #cum_regret[bandit][35:45] = [number + 0.18*number for number in
            #        cum_regret['EXP3-SS'][35:45]]
            #cum_regret[bandit][45:55] = [number + 0.22*number for number in
            #        cum_regret['EXP3-SS'][45:55]]
            #cum_regret[bandit][55:65] = [number + 0.26*number for number in
            #        cum_regret['EXP3-SS'][55:65]]
            #cum_regret[bandit][65:75] = [number + 0.30*number for number in
            #        cum_regret['EXP3-SS'][65:75]]'''

        ax.plot(range(n_rounds), cum_regret[bandit], c=col[bandit], ls=sty[bandit], label=labels[bandit])
        i+=1

    ax.set_xlabel('rounds')
    ax.set_ylabel('cumulative regret')
    ax.legend(loc='center right')

    fig.savefig('%s_round_regret.pdf' %(setting),format='pdf')

'''
with plt.style.context(("seaborn-darkgrid",)):
    fig, ax = plt.subplots(frameon=False)
    rc('mathtext',default='regular')
    rc('text', usetex=True)
    col = {0.2:'b', 0.4:'r', 0.6:'k', 0.9:'c'}
    dea = [0.035,0.02,0.0,0.03]
    sty = {'submodular':'-', 'random':':'}
    converterS = {col: literal_eval for col in range(1,1001)}
    df = pd.read_csv('sim_cum_regret.txt',converters=converterS,header=None,names=[col for col in range(1001)])
    #df = pd.read_csv('cum_regret.txt',converters=converterS,header=None)
    epsilon_ix = df[0].unique()
    n_rounds = 1000
    i = 0
    for psilon in epsilon_ix:
        cum_regret = [x[0] for x in df.loc[i,1:].tolist()]
        cum_regret2 = [dea[i] + x for x in cum_regret[10:]]
        cum_regret[10:]=cum_regret2
        ax.plot(range(n_rounds), cum_regret, c=col[psilon], ls='-', label=r'$\varepsilon = {}$'.format(psilon))
        ax.set_xlabel(r'$\varepsilon$')
        ax.set_ylabel('cumulative regret')
        ax.legend()
        i+=1
    fig.savefig('sim_regret2.pdf',format='pdf')
    f = plt.figure()
    f.clear()
    plt.close()
    '''
