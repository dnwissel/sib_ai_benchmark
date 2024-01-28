import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from pathlib import Path
import os
import pickle


def subplot(data, ax, info, top=1.15, bottom=0.5):
    # def subplot(data, ax, info, top=0.15, bottom=-0.05):
    bp = ax.boxplot(data, notch=False, sym='+', vert=True, whis=1.5)
    # fig.canvas.manager.set_window_title('A Boxplot Example')

    # bp = ax.boxplot(data, notch=False, sym='+', vert=True, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')
    # plt.show()

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                  alpha=0.5)

    ylabel = info['metric_name'].upper()
    ylabel = ylabel.replace('_', ' ')
    ax.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title=f'{info["tissue_name"].upper()}',
        xlabel=None,
        # xlabel='Model',
        ylabel=ylabel
    )
    # Now fill the boxes with desired colors
    box_colors = ['darkkhaki', 'royalblue']
    num_boxes = len(data)
    medians = np.empty(num_boxes)
    for i in range(num_boxes):
        box = bp['boxes'][i]
        box_x = []
        box_y = []
        for j in range(5):
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])
        box_coords = np.column_stack([box_x, box_y])
        # Alternate between Dark Khaki and Royal Blue
        ax.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        median_x = []
        median_y = []
        for j in range(2):
            median_x.append(med.get_xdata()[j])
            median_y.append(med.get_ydata()[j])
            ax.plot(median_x, median_y, 'k')
        medians[i] = median_y[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax.plot(np.average(med.get_xdata()), np.average(data[i]),
                color='w', marker='*', markeredgecolor='k')

    # Set the axes ranges and axes labels
    ax.set_xlim(0.5, num_boxes + 0.5)
    ax.set_ylim(bottom, top)
    ax.set_xticklabels(info['labels'], rotation=45, fontsize=8)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(round(s, 4)) for s in medians]
    weights = ['bold', 'semibold']
    for tick, label in zip(range(num_boxes), ax.get_xticklabels()):
        k = tick % 2
        ax.text(pos[tick], .90, upper_labels[tick],
                transform=ax.get_xaxis_transform(),
                horizontalalignment='center', size='x-small',
                weight=weights[k], color=box_colors[k])

    # # Finally, add a basic legend
    # fig.text(0.80, 0.08, f'{N} Random Numbers',
    #         backgroundcolor=box_colors[0], color='black', weight='roman',
    #         size='x-small')
    # fig.text(0.80, 0.045, 'IID Bootstrap Resample',
    #         backgroundcolor=box_colors[1],
    #         color='white', weight='roman', size='x-small')
    # fig.text(0.80, 0.015, '*', color='white', backgroundcolor='silver',
    #         weight='roman', size='medium')
    # fig.text(0.815, 0.013, ' Average Value', color='black', weight='roman',
    #         size='x-small')


def plot(results, metric_name, path, ncols=2):
    # Prepare data
    data = []
    info = {}

    tissue_names = list(results['datasets'].keys())
    tissue_names = sorted(tissue_names)

    model_names = results['datasets'][tissue_names[0]]['model_results'].keys()
    model_names = sorted(model_names)

    info['labels'] = list(model_names)
    info['metric_name'] = metric_name

    # print(info['labels'])
    for tn in tissue_names:
        res_tissue = results['datasets'][tn]['model_results']
        res_model = []
        for mn in model_names:
            # TODO: refactor benchmark
            if metric_name in ['ece', 'ece_uc']:
                res_model.append(res_tissue[mn][metric_name])
            else:
                res_model.append(res_tissue[mn]['scores'][metric_name]['full'])
        data.append(res_model)

    # Plot
    nrows = len(tissue_names) // ncols + 1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 12))
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    # print(axs.shape)
    cnt = 0
    for row_idx in range(axs.shape[0]):
        if len(axs.shape) > 1:
            for col_idx in range(axs.shape[1]):
                if cnt < len(tissue_names):
                    info['tissue_name'] = tissue_names[cnt]
                    subplot(data[cnt], axs[row_idx, col_idx], info)
                else:
                    axs[row_idx, col_idx].axis('off')
                cnt += 1
        else:
            if cnt < len(tissue_names):
                info['tissue_name'] = tissue_names[cnt]
                subplot(data[cnt], axs[row_idx], info)
            else:
                axs[row_idx].axis('off')
            cnt += 1

    # configure x/y labels
    # for ax in axs.flat:
    #     ax.label_outer()
    plt.subplots_adjust(wspace=0.25, hspace=0.8)
    plt.savefig(path, dpi=450)
    # plt.show()


def load_res(path):
    fns = []
    for fn in os.listdir(path):
        if 'pkl' in fn:
            fns.append(fn)
    return fns


if __name__ == "__main__":
    parent_path = Path(__file__).parents[2]
    path_res = os.path.join(parent_path, 'results/flat')
    path_res = os.path.join(parent_path, 'results/hier')

    # metric_name = 'balanced_accuracy_score'
    metric_name = 'accuracy'
    metric_name = 'ece'
    metric_name = 'F1_SCORE_MACRO'.lower()
    metric_name = 'f1_hier'

    fns = load_res(path_res)
    fns = ['scanvi_bcm_flat.pkl']
    fns = ['scanvi_bcm_global_local_path-eval.pkl']
    print(fns)

    for fn in fns:
        # if 'path-eval' in fn  or 'global' in fn:
        # continue

        with open(path_res + f'/{fn}', 'rb') as fh:
            results = pickle.load(fh)
        fn = fn[:-4] + '.pdf'
        plot(results, metric_name, path_res + f'/plots/{metric_name}/{fn}')
