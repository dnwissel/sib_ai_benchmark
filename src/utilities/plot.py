import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


def subplot(data, ax, info):
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

    ax.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title=f'Tissue {info["tissue_name"]}',
        xlabel='Model',
        ylabel=info['metric_name']
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
    top = 1.08
    bottom = -0.05
    ax.set_ylim(bottom, top)
    ax.set_xticklabels(info['labels'], rotation=45, fontsize=8)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(round(s, 2)) for s in medians]
    weights = ['bold', 'semibold']
    for tick, label in zip(range(num_boxes), ax.get_xticklabels()):
        k = tick % 2
        ax.text(pos[tick], .95, upper_labels[tick],
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


def plot(results, metric_name, task_name):
    # Prepare data
    data = []
    info = {}
    tissue_names = list(results['datasets'].keys())
    model_names = results['datasets'][tissue_names[0]]['model_results'].keys()

    info['labels'] = list(model_names)
    info['metric_name'] = metric_name

    print(info['labels'])
    for tn in tissue_names:
        res_tissue = results['datasets'][tn]['model_results']
        res_model =[]
        for mn in model_names:
            res_model.append(res_tissue[mn][metric_name]['full'])
        data.append(res_model)

    # Plot
    ncols = 2
    nrows = len(tissue_names) // ncols + 1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 12))
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    print(axs.shape)
    cnt = 0
    for row_idx in range(axs.shape[0]):
        for col_idx in range(axs.shape[1]):
            if cnt < len(tissue_names):
                info['tissue_name'] = tissue_names[cnt]
                subplot(data[cnt], axs[row_idx, col_idx], info)
            else:
                axs[row_idx, col_idx].axis('off')
            cnt += 1 

    # configure x/y labels
    for ax in axs.flat:
        ax.label_outer()
    
    plt.savefig(task_name)
    # plt.show()


if __name__ == "__main__":
    results = {'datasets': {
                        'head': 
                        {'model_results': 
                            {'nn': 
                                {'F1': [0.95, 0.86, 0.63, 1.0]
                                },
                            'LR': 
                                {'F1': [0.95, 0.86, 0.63, 1.0]
                                }
                            }
                            },
                        
                        'antenna': 
                        {'model_results': 
                            {'nn': 
                                {'F1': [0.95, 0.86, 0.63, 1.0]
                                },
                            'LR': 
                                {'F1': [0.95, 0.86, 0.63, 1.0]
                                }
                        }
                            },
                        'head1': 
                        {'model_results': 
                            {'nn': 
                                {'F1': [0.95, 0.86, 0.63, 1.0]
                                },
                            'LR': 
                                {'F1': [0.95, 0.86, 0.63, 1.0]
                                }
                            },
                        
                        'antenna1': 
                        {'model_results': 
                            {'nn': 
                                {'F1': [0.95, 0.86, 0.63, 1.0]
                                },
                            
                            'LR': 
                                {'F1': [0.95, 0.86, 0.63, 1.0]
                                }
                            }
                        }
                        }}        
                }

    metric_name = 'F1'
    plot(results, metric_name, 'test')
