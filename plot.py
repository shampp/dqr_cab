import matplotlib.pyplot as plt
from matplotlib import rc

def simple_plot(n_rounds, noof_anchors, data, plot_fname, raw_fname, xlab, ylab):
    if (not isinstance(data,dict)):
        print("data should be dictionary")
        exit(1)

    colors = ['b', 'r', 'k', 'c', 'g', 'c', 'm', 'y']
    f = plt.figure()
    f.clear()
    plt.clf()
    plt.close(f)
    with plt.style.context(("seaborn-darkgrid",)):
        fig, ax = plt.subplots(frameon=False)
        rc('mathtext',default='regular')
        rc('text', usetex=True)
        candidate_ix = data.keys()
        col = {}
        for i,ix in enumarate(candidate_ix):
            col[ix] = colors[i] #{2:'b', 3:'r', 5:'k', 10:'c'}

        with open(raw_fname, "w") as raw_fd:
            for cand_sz in candidate_ix:
                cum_regret = [sum(x)/noof_anchors for x in zip(*data[cand_sz].values())]
                val = str(cand_sz)+','+','.join([str(e) for e in cum_regret])
                print(val, file=raw_fd)
                ax.plot(range(n_rounds), cum_regret, c=col[cand_sz], ls='-', label=r'{}'.format(cand_sz))
                ax.set_xlabel(r'{}'.format(xlab))
                ax.set_ylabel(r'{}'.format(ylab))
                ax.legend()
            fig.savefig(plot_fname,format='pdf')
            f = plt.figure()
            f.clear()
            plt.close(f)


def twinx_plot(n_rounds, noof_anchors, data1, data2, plot_fname, raw_fname, xlab, ylab1, ylab2):
    if (type(data1) != type(data2)):
        print("data type of the plot data differs ..... Exiting")
        exit(1)
    if (not isinstance(data1,dict)):
        print("data should be dictionary")
        exit(1)

    colors = ['b', 'r', 'k', 'c', 'g', 'c', 'm', 'y']
    f = plt.figure()
    f.clear()
    plt.clf()
    plt.close(f)
    with plt.style.context(("seaborn-darkgrid",)):
        fig, ax1 = plt.subplots(frameon=False)
        rc('mathtext',default='regular')
        rc('text', usetex=True)
        candidate_ix = data1.keys()
        col = {}
        for i,ix in enumarate(candidate_ix):
            col[ix] = colors[i] #{2:'b', 3:'r', 5:'k', 10:'c'}

        with open(raw_fname, "w") as raw_fd:
            for key in candidate_ix:
                d_vals = [sum(x)/noof_anchors for x in zip(*data1[key].values())]
                val = str(key)+','+','.join([str(e) for e in d_vals])
                print(val, file=raw_fd)
                ax1.plot(range(n_rounds), d_vals, c=col[key], ls='-',
                        label=r'{}'.format(key))

            ax1.set_ylabel(r'{}'.format(ylab1))
            ax1.legend()
            ax2 = ax1.twinx();
        with open(raw_fname, "a") as raw_fd:
            for key in candidate_ix:
                d_vals = [sum(x)/noof_anchors for x in zip(*data2[key].values())]
                val = str(key)+','+','.join([str(e) for e in d_vals])
                print(val, file=raw_fd)
                ax2.plot(range(n_rounds), d_vals, c=col[key], ls='-',
                        label=r'{}'.format(key))
            ax2.set_ylabel(r'{}'.format(ylab2))

            ax1.set_xlabel(r'{}'.format(xlab))
            fig.savefig(plot_fname,format='pdf')
            f = plt.figure()
            f.clear()
            plt.close(f)
