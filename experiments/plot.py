import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

sns.set_theme()
sns.set(font_scale=1.5)
sns.set_style('whitegrid',
              {"font.family": 'Microsoft YaHei',
               "font.sans-serif": 'Microsoft YaHei',
               "axes.unicode_minus": False})


def add_cumulative_time(df: pd.DataFrame, name='time'):
    df[name] = df['time'].cumsum()


def long_form_of_time(dfs, names):
    parts = []
    for df, name in zip(dfs, names):
        view = df.loc[:, ['time']]
        view['n'] = np.arange(1, view.shape[0] + 1)
        view['method'] = name
        parts.append(view)
    return pd.concat(parts)


def long_form_of_metrics(dfs, names):
    parts = []
    for df, name in zip(dfs, names):
        view = df.loc[:, ['iou', 'precision', 'recall']]
        view = view.melt(var_name='metrics', value_name='value')
        view['method'] = name
        parts.append(view)
    return pd.concat(parts)


def rect_vec_time_plot(save=False):
    prefix = Path('result/rect_vec')
    pca_min_max = pd.read_csv(prefix / 'pca_min_max.csv')
    pca_gaussian = pd.read_csv(prefix / 'pca_gaussian.csv')
    softras = pd.read_csv(prefix / 'softras.csv').iloc[-716:]
    for df in [pca_min_max, pca_gaussian, softras]:
        add_cumulative_time(df)

    long_data = long_form_of_time([pca_min_max, pca_gaussian, softras],
                                  names=['pca-min-max', 'pca-gaussian', 'softras'])
    long_data_pca = long_data[long_data.method != 'softras']
    long_data_softras = long_data[long_data.method == 'softras']

    # create two axes that share x
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
    fig.subplots_adjust(hspace=0.05)

    # plot the same data
    style_kws = {'linewidth': 5, 'alpha': 0.7, 'marker': 'o', 'markevery': 50}
    sns.lineplot(data=long_data_softras, x='n', y='time', hue='method',
                 hue_order=['pca-min-max', 'pca-gaussian', 'softras'], ax=ax1, **style_kws)
    sns.lineplot(data=long_data_pca, x='n', y='time', hue='method', ax=ax2, **style_kws)

    # set scale separately
    ax1.set_yscale('symlog', base=10)
    ax2.set_yscale('linear')

    # manually set cut point
    ax1.set_ylim(1500, 5E7)
    ax2.set_ylim(None, 1500)

    # hide axes' spines in the middle
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    # create slanted cut-out lines
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    # modify legend
    ax1.legend(bbox_to_anchor=[0.35, 1])
    ax2.get_legend().remove()

    # modify axis labels
    ax1.set_ylabel('')
    ax2.set_ylabel("耗时 / ms")
    ax2.yaxis.set_label_coords(-0.1, 1.1)
    ax2.set_xlabel('数量')
    plt.tight_layout()

    if save:
        plt.savefig('result/figures/rect_vec_time.pdf')

    plt.show()


def rect_vec_metric_plot(save=False):
    prefix = Path('result/rect_vec')
    pca_min_max = pd.read_csv(prefix / 'pca_min_max.csv')
    pca_gaussian = pd.read_csv(prefix / 'pca_gaussian.csv')
    softras = pd.read_csv(prefix / 'softras.csv')
    long_data = long_form_of_metrics([pca_min_max, pca_gaussian, softras],
                                     names=['pca-min-max', 'pca-gaussian', 'softras'])

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    sns.barplot(data=long_data,
                x='metrics', y='value', hue='method',
                ci='sd', capsize=0.1, ax=ax)
    plt.xlabel('指标类型')
    plt.ylabel('值')
    ax.legend(bbox_to_anchor=(0.35, 1.))
    plt.tight_layout()

    if save:
        plt.savefig('result/figures/rect_vec_metrics.pdf')
    plt.show()
    pass


def room_vec_metric_plot(save=False):
    prefix = Path('result/room_contour')

    dfs = []
    names = []
    for p in prefix.iterdir():
        if p.is_file() and p.suffix == '.csv':
            name = p.name.rsplit('.', 1)[0].replace(',', '/')
            locals()[name] = pd.read_csv(p)
            dfs.append(locals()[name])
            names.append(name)
    long_data = long_form_of_metrics(dfs, names=names)

    sns.set_palette('light:b', 8)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    sns.barplot(data=long_data, x='metrics', y='value',
                hue='method', hue_order=['w/o', 'w/iou', 'w/boundary', 'w/orth', 'w/iou+boundary', 'w/iou+orth', 'w/boundary+orth','full implement'],
                ci='sd', capsize=0.,
                ax=ax)
    plt.legend(ncol=4, bbox_to_anchor=(0.98, 1.2),)
    plt.ylim(0, None)
    plt.xlabel('指标类型')
    plt.ylabel('值')
    plt.tight_layout()
    if save:
        plt.savefig('result/figures/room_vec_metrics.pdf')
    plt.show()


def wcl_vec_metric_plot(save=False):
    prefix = Path('result/wcl')

    dfs = []
    names = []
    for p in prefix.iterdir():
        if p.is_file() and p.suffix == '.csv':
            name = p.name.rsplit('.', 1)[0].replace(',', '/')
            locals()[name] = pd.read_csv(p)
            dfs.append(locals()[name])
            names.append(name)
    long_data = long_form_of_metrics(dfs, names=names)

    sns.set_palette('light:orange', 8)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    sns.barplot(data=long_data, x='metrics', y='value',
                hue='method', hue_order=['w/o', 'w/center', 'w/nearby', 'w/alignment', 'w/center+nearby', 'w/center+alignment', 'w/nearby+alignment', 'full implement'],
                ci='sd', capsize=0,
                ax=ax)
    plt.ylim(0.5, 0.95)
    plt.legend(ncol=4, bbox_to_anchor=(1.06, 1.2), )
    plt.xlabel('指标类型')
    plt.ylabel('值')
    plt.tight_layout()

    if save:
        plt.savefig('result/figures/wcl_vec_metrics.pdf')

    plt.show()


if __name__ == "__main__":
    # rect_vec_time_plot(save=False)
    # rect_vec_metric_plot(save=False)
    # room_vec_metric_plot(save=False)
    wcl_vec_metric_plot(save=False)
    pass
