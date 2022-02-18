import matplotlib.pyplot as plt
from matplotlib import rc

def simple_plot(data,filename):
    f = plt.figure()
    f.clear()
    plt.clf()
    plt.close(f)
    with plt.style.context(("seaborn-darkgrid",)):
        fig, ax = plt.subplots(frameon=False)
        rc('mathtext',default='regular')
        rc('text', usetex=True)
        col = {2:'b', 3:'r', 5:'k', 10:'c'}
        regret_file = 'cand_cum_regret.txt'
        with open(regret_file, "w") as regret_fd:
            for cand_sz in candidate_ix:
                cum_regret = [sum(x)/noof_anchors for x in zip(*regret[cand_sz].values())]
                val = str(cand_sz)+','+','.join([str(e) for e in cum_regret])
                print(val, file=regret_fd)
                ax.plot(range(n_rounds), cum_regret, c=col[cand_sz], ls='-', label=r'$k = {}$'.format(cand_sz))
                ax.set_xlabel(r'k')
                ax.set_ylabel(r'cumulative regret')
                ax.legend()
            fig.savefig('arm_regret_%s.pdf' %(setting),format='pdf')
            f = plt.figure()
            f.clear()
            plt.close(f)

def twinx_plot(data1,data2,filename):
    if (type(data1) != type(data2)):
        print("data type of the plot data differs ..... Exiting")
        exit(1)
    if (not isinstance(data1,dict)):
        print("data should be dictionary")
        exit(1)
             
    f = plt.figure()
    f.clear()
    plt.clf()
    plt.close(f)
    with plt.style.context(("seaborn-darkgrid",)):
        fig, ax1 = plt.subplots(frameon=False)
        rc('mathtext',default='regular')
        rc('text', usetex=True)
        col = {2:'b', 3:'r', 5:'k', 10:'c'}
        sim_file = 'cand_cum_sim.txt'
        with open(regret_file, "w") as regret_fd:
            for key in data1:
                cum_regret = [sum(x)/noof_anchors for x in zip(*regret[cand_sz].values())]
                val = str(cand_sz)+','+','.join([str(e) for e in cum_regret])
                print(val, file=regret_fd)
                ax1.plot(range(n_rounds), cum_regret, c=col[cand_sz], ls='-', label=r'$k = {}$'.format(cand_sz))

            ax1.set_ylabel(r'cumulative regret')
            ax1.legend()
            ax2 = ax1.twinx();
        with open(regret_file, "w") as regret_fd:
            for key in data1:
                cum_regret = [sum(x)/noof_anchors for x in zip(*regret[cand_sz].values())]
                val = str(cand_sz)+','+','.join([str(e) for e in cum_regret])
                print(val, file=regret_fd)
                ax2.plot(range(n_rounds), cum_regret, c=col[cand_sz], ls='-', label=r'$k = {}$'.format(cand_sz))
                ax2.set_ylabel(r'cumulative regret')

            ax1.set_xlabel(r'k')
            fig.savefig('arm_regret_%s.pdf' %(setting),format='pdf')
            f = plt.figure()
            f.clear()
            plt.close(f)
