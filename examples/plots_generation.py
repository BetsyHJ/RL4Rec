#%%
from os import name
import sys
from matplotlib.pyplot import xticks, ylabel, ylim
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pylab
from matplotlib import rcParams
matplotlib.rcParams['text.usetex'] = True

seeds = [x for x in range(2012, 2022)]
dataset = 'yahoo' #'sim4' or 'coat' or 'yahoo'
if len(sys.argv) == 2:
    dataset = sys.argv[1]

lines = {}
line_colors = sns.color_palette()
line_colors[-1], line_colors[-4] = line_colors[-4], line_colors[-1]
line_styles = ['--', ':', '-.', '-', '-', '-', '-', '--', ':', '-.']
line_markers = ['o', 'o', 'o', 'o', 's', 'P', 'D', 's', 's', 's']
for i, mode in enumerate(['boi', 'pld', 'mlp', 'att', 'mlp_tanh', 'cnn', 'gru', 'mlp_relu', 'mlp_sigmoid']):
    lines[mode] = [line_styles[i], line_markers[i], line_colors[i]]


def draw(path, folder='result/', n_points = None):
    df = []
    for seed in seeds:
        file_name = folder + path % seed
        df_ = pd.read_csv(file_name, header=None, index_col=False, sep=' ')
        if n_points:
            df_ = df_[df_.index % n_points == 0]
        df_.columns = ['step', 'result']
        df_['seed'] = seed
        df.append(df_)
    return pd.concat(df)


# %%
# # learning curves
if dataset == 'coat':
    file_att = "coat_DQN_Att_seed%d_acdim_64_rnndim32plot.data"
    file_cnn = "coat_DQN_CNN_seed%d_acdim_64_rnndim64plot.data"
    file_gru = "coat_DQN_GRU_seed%d_acdim_32_rnndim16plot.data"
    file_mlp = "coat_DQN_MLP_seed%d_acdim_64plot.data"
    file_mlp_tanh = "coat_DQN_MLP_seed%d_acdim_64_tanhplot.data"
    file_boi = "coat_DQN_BOI_seed%d_acdim_64plot.data"
    file_pld = "coat_DQN_PLD_seed%d_acdim_64plot.data"
else:
    file_att = "yahoo_DQN_Att_seed%d_acdim_64_rnndim16plot.data" 
    file_cnn = "yahoo_DQN_CNN_seed%d_acdim_64_rnndim32plot.data"
    file_gru = "yahoo_DQN_GRU_seed%d_acdim_32_rnndim32plot.data"
    file_mlp = "yahoo_DQN_MLP_seed%d_acdim_64plot.data"
    file_mlp_tanh = "yahoo_DQN_MLP_seed%d_acdim_64_reluplot.data"
    file_boi = "yahoo_DQN_BOI_seed%d_acdim_64plot.data"
    file_pld = "yahoo_DQN_PLD_seed%d_acdim_64plot.data"

folder = 'learning_curves/'
n_points = 3
if dataset == 'yahoo':
    n_points = 10

df2_att = draw(file_att, folder=folder, n_points=n_points)
df2_att['mode'] = 'att'

df2_cnn = draw(file_cnn, folder=folder, n_points=n_points)
df2_cnn['mode'] = 'cnn'

df2_gru = draw(file_gru, folder=folder, n_points=n_points)
df2_gru['mode'] = 'gru'

df2_mlp = draw(file_mlp, folder=folder, n_points=n_points)
df2_mlp['mode'] = 'mlp'

df2_mlp_tanh = draw(file_mlp_tanh, folder=folder, n_points=n_points)
df2_mlp_tanh['mode'] = 'mlp_tanh'

df2_boi = draw(file_boi, folder=folder, n_points=n_points)
df2_boi['mode'] = 'boi'

df2_pld = draw(file_pld, folder=folder, n_points=n_points)
df2_pld['mode'] = 'pld'

df2 = pd.concat([df2_att, df2_cnn, df2_gru, df2_mlp, df2_mlp_tanh, df2_boi, df2_pld])
df2['step'] /= 1000.0

#%%
rcParams['figure.figsize'] = 5, 5 # 8, 3 # 6, 2
def draw2(df2, hue_order=['boi', 'pld', 'mlp', 'att', 'mlp_tanh', 'cnn', 'gru'], names=['BOI', 'PLD', 'Avg', 'Attention', 'MLP', 'CNN', 'GRU'], filename='', ylim=(0.4, 3.0), ax=None):
    palette = [lines[x][2] for x in hue_order]
    plot2 = sns.lineplot(data=df2, x='step', y='result', hue='mode', hue_order=hue_order, palette=palette, ax=ax)
    for i, mode in enumerate(hue_order):
        plot2.lines[i].set_linestyle(lines[mode][0])
        plot2.lines[i].set_marker(lines[mode][1])
        plot2.lines[i].set_markevery(7)
        if lines[mode][1] == 'P':
            plot2.lines[i].set_markersize(8)
    figlegend = pylab.figure()
    figlegend.legend(plot2.lines, names, 
                fontsize=18,
                loc='center',
                ncol=7,
                frameon=False,
                borderaxespad=0,
                borderpad=0,
                labelspacing=0.2,
                columnspacing=1., 
                markerscale=1.25)
    # figlegend.show()
    figlegend.savefig('train_%s%s_legend.pdf' % (dataset, filename), bbox_inches='tight', pad_inches=0)
    plt.close(figlegend)
    plot2.legend_.remove()
    plot2.tick_params(right=True)
    if dataset == 'coat':
        plot2.set(xlim=(0, 350), ylim=ylim, xlabel='training step (thousand)', ylabel='cumulative number of clicks')
    else:
        plot2.set(xlim=(0, 1000), ylim=ylim, xlabel='training step (thousand)', ylabel='cumulative number of clicks')
    if ax is None:
        plot2.figure.savefig('train_%s%s.pdf' % (dataset, filename), bbox_inches='tight', pad_inches=0)

# draw2(df2, ['boi', 'pld', 'mlp', 'att', 'mlp_tanh', 'cnn', 'gru'], ['BOI', 'PLD', 'Avg', 'Attention', 'MLP', 'CNN', 'GRU'])
fig, axs = plt.subplots(2, sharex=True)
plt.subplots_adjust(hspace = 0.1)
if dataset == 'coat':
    ylim1, ylim2 = (0.4, 3.0), (0.4, 2.5)
elif dataset == 'yahoo':
    ylim1, ylim2 = (1.0, 1.42), (1.0, 1.42)
draw2(df2, hue_order=['att', 'boi', 'pld', 'mlp'], names=['Attention', 'BOI', 'PLD', 'Avg'], filename='_four', ylim=ylim2, ax=axs[0])
draw2(df2, hue_order=['mlp_tanh', 'cnn', 'gru', 'att'], names=['MLP', 'CNN', 'GRU', 'Attention'], filename='_addition', ylim=ylim1, ax=axs[1]) # here mlp means no tanh and renamed as Avg, and mlp_tanh is the new mlp.
plt.savefig('train_%s.pdf' % dataset, bbox_inches='tight', pad_inches=0)


#%%
# # results
if dataset == 'coat':
    file_att = "coat_DQN_Att_seed%d_acdim_64_rnndim32_eval.txt"
    file_cnn = "coat_DQN_CNN_seed%d_acdim_64_rnndim64_eval.txt"
    file_gru = "coat_DQN_GRU_seed%d_acdim_32_rnndim16_eval.txt"
    file_mlp = "coat_DQN_MLP_seed%d_acdim_64_eval.txt"
    file_mlp_tanh = "coat_DQN_MLP_seed%d_acdim_64_tanh_eval.txt"
    file_mlp_relu = "coat_DQN_MLP_seed%d_acdim_64_relu_eval.txt"
    file_mlp_sigmoid = "coat_DQN_MLP_seed%d_acdim_64_sigmoid_eval.txt"
    file_boi = "coat_DQN_BOI_seed%d_acdim_64_eval.txt"
    file_pld = "coat_DQN_PLD_seed%d_acdim_64_eval.txt"
else:
    file_att = "yahoo_DQN_Att_seed%d_acdim_64_rnndim16_eval.txt"
    file_cnn = "yahoo_DQN_CNN_seed%d_acdim_64_rnndim32_eval.txt"
    file_gru = "yahoo_DQN_GRU_seed%d_acdim_32_rnndim32_eval.txt"
    file_mlp = "yahoo_DQN_MLP_seed%d_acdim_64_eval.txt"
    file_mlp_tanh = "yahoo_DQN_MLP_seed%d_acdim_64_tanh_eval.txt"
    file_mlp_relu = "yahoo_DQN_MLP_seed%d_acdim_64_relu_eval.txt"
    file_mlp_sigmoid = "yahoo_DQN_MLP_seed%d_acdim_64_sigmoid_eval.txt"
    file_boi = "yahoo_DQN_BOI_seed%d_acdim_64_eval.txt"
    file_pld = "yahoo_DQN_PLD_seed%d_acdim_64_eval.txt"

df_att = draw(file_att)
df_att['mode'] = 'att'

df_cnn = draw(file_cnn)
df_cnn['mode'] = 'cnn'

df_gru = draw(file_gru)
df_gru['mode'] = 'gru'

df_mlp = draw(file_mlp)
df_mlp['mode'] = 'mlp'

df_mlp_tanh = draw(file_mlp_tanh)
df_mlp_tanh['mode'] = 'mlp_tanh'

df_mlp_relu = draw(file_mlp_relu)
df_mlp_relu['mode'] = 'mlp_relu'

df_mlp_sigmoid = draw(file_mlp_sigmoid)
df_mlp_sigmoid['mode'] = 'mlp_sigmoid'

df_boi = draw(file_boi)
df_boi['mode'] = 'boi'

df_pld = draw(file_pld)
df_pld['mode'] = 'pld'

df = pd.concat([df_att, df_cnn, df_gru, df_mlp, df_mlp_tanh, df_boi, df_pld, df_mlp_relu, df_mlp_sigmoid])

#%%
plt.close()
matplotlib.rcParams['figure.figsize'] = 5, 5.5

def draw1(df, hue_order=['mlp', 'cnn', 'gru', 'att', 'boi', 'pld', 'mlp_tanh'], names=['Avg', 'CNN', 'GRU', 'Attention', 'BOI', 'PLD', 'MLP'], filename='', ylim=(0.0, 3.1), yticks=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], ax=None):
    # get colors
    palette = [lines[x][2] for x in hue_order]
    plot1 = sns.lineplot(data=df, x='step', y='result', hue='mode', hue_order=hue_order, palette=palette, ax=ax) # ci=95 default
    for i, mode in enumerate(hue_order):
        plot1.lines[i].set_linestyle(lines[mode][0])
        plot1.lines[i].set_marker(lines[mode][1])
        if lines[mode][1] == 'P':
            plot1.lines[i].set_markersize(8)
        # plot1.lines[i].set_fillstyle('none')
    figlegend = pylab.figure()
    figlegend.legend(plot1.lines, names, 
                fontsize=18,
                loc='center',
                ncol=7,
                frameon=False,
                borderaxespad=0,
                borderpad=0,
                labelspacing=0.2,
                columnspacing=1., 
                markerscale=1.25)
    # figlegend.show()
    figlegend.savefig('eval_%s%s_legend.pdf' % (dataset, filename), bbox_inches='tight', pad_inches=0)
    plt.close(figlegend)
    plot1.legend_.remove()
    plot1.tick_params(right=True)
    if dataset == 'coat':
        plot1.set(xlim=(1, 10), ylim=ylim, yticks=yticks, xlabel='recommendation turn', ylabel='cumulative number of clicks')
    else:
        plot1.set(xlim=(1, 10), ylim=(0.1, 1.24), xlabel='recommendation turn', ylabel='cumulative number of clicks')
    if ax is None:
        plot1.figure.savefig('eval_%s%s.pdf' % (dataset, filename), bbox_inches='tight', pad_inches=0)
# draw1(df, ['boi', 'pld', 'mlp', 'att', 'mlp_tanh', 'cnn', 'gru'], ['BOI', 'PLD', 'Avg', 'Attention', 'MLP', 'CNN', 'GRU'])
fig, axs = plt.subplots(2, sharex=True)
plt.subplots_adjust(hspace = 0.1)
draw1(df, hue_order=['att', 'boi', 'pld', 'mlp'], names=['Attention', 'BOI', 'PLD', 'Avg'], filename='_linear', ylim=(0, 2.6), yticks=[0, 0.5, 1, 1.5, 2, 2.5], ax=axs[0])
if dataset == 'yahoo':
    lines['mlp_relu'] = [line_styles[4], line_markers[4], line_colors[4]] # keep the line of relu the same as tanh
    draw1(df, hue_order=['mlp_relu', 'cnn', 'gru', 'att'], names=['MLP', 'CNN', 'GRU', 'Attention'], filename='_nonlinear', ax=axs[1])
else:
    draw1(df, hue_order=['mlp_tanh', 'cnn', 'gru', 'att'], names=['MLP', 'CNN', 'GRU', 'Attention'], filename='_nonlinear', ax=axs[1]) # here mlp means no tanh and renamed as Avg, and mlp_tanh is the new mlp.
plt.savefig('eval_%s.pdf' % dataset, bbox_inches='tight', pad_inches=0)

# # for mlp with different activation functions
plt.close()
# rcParams['figure.figsize'] = 5, 2.5
matplotlib.rcParams['figure.figsize'] = 5, 2.5
lines['mlp_tanh'] = [line_styles[9], line_markers[9], line_colors[9]] 
draw1(df, ['mlp', 'mlp_tanh', 'mlp_relu', 'mlp_sigmoid'], ['Avg', 'MLP with tanh', 'MLP with relu', 'MLP with sigmoid'], filename="_MLPs")

