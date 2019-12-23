from sto_reg.tests.net_training.train import Trainer
import numpy as np
from bokeh.plotting import figure, output_file, show
from bokeh.models import Range1d, Legend, ColumnDataSource, FactorRange
import argparse
from itertools import cycle
import torch
from sto_reg.tests.net_training.cnn import LeNet5
from bokeh.transform import factor_cmap
from bokeh.palettes import RdYlGn5


parser = argparse.ArgumentParser(description='PyTorch MNIST')
parser.add_argument('--reg_type', type=str, default='SO', metavar='reg',
                    help='Regularization type, one of DO, SO or BO')
args = parser.parse_args()


def get_experiment_configs(reg_type):
    experiments = []
    if reg_type == 'DO':
        experiments += [{'name':'LeNet5_'+reg_type+'_Conv', 'drop_rate':0.3},
        {'name':'LeNet5_'+reg_type+'_Fc', 'drop_rate':0.3},
        {'name':'LeNet5_'+reg_type+'_All', 'drop_rate':0.3},]
    else:        
        for q in [1.5, 1.75, 2.0, 2.25, 2.5]:
            experiments += [{'name':'LeNet5_'+reg_type+'_Conv', 'drop_rate':0.3, 'bo_norm':q},
            {'name':'LeNet5_'+reg_type+'_Fc', 'drop_rate':0.3, 'bo_norm':q},
            {'name':'LeNet5_'+reg_type+'_All', 'drop_rate':0.3, 'bo_norm':q},]
    return experiments

def train_networks():
    archs = get_experiment_configs(args.reg_type)
    for arch in archs:
        t = Trainer(arch)
        t.run(256)

def plot_accuracies():
    pbo = figure(plot_width=1400, plot_height=500)
    pbo.title.text = 'LeNet5 trained on MNIST with different regularizations'
    dash = cycle(['solid', 'dashed', 'dotted', 'dotdash', 'dashdot'])
    color = cycle(['red','blue','green','black','orange', 'cyan', 'purple','magenta','indigo'])
    metric = 'validation_accuracy'
    max_acc = {}
    legend_it=[]
    for reg_type in ['SO', 'BO', 'DO']:
        archs = get_experiment_configs(reg_type)
        for arch in archs:
            path = 'net_training/trained_data/'
            for k,v in arch.items():
                path += str(k) + '_' + str(v) + '_'
            data = np.genfromtxt(path+metric, delimiter=',')
            if reg_type == 'DO':
                label = arch['name'][7:] + '_' + str(arch['drop_rate'])
            else:
                label = arch['name'][7:] + '_' + str(arch['drop_rate']) + '_' + str(arch['bo_norm'])
            max_acc[label] = np.max(data)
            c = pbo.line(range(len(data)),data, line_width=5, line_dash= next(dash), color=next(color), alpha=0.8)
            legend_it.append((label,[c]))
    
    half = len(legend_it)//2
    legend1 = Legend(items=legend_it[:half], location=(0, 0))
    legend1.click_policy="hide"
    pbo.add_layout(legend1, 'left')
    
    legend1 = Legend(items=legend_it[half:], location=(0, -10))
    legend1.click_policy="hide"
    pbo.add_layout(legend1, 'right')
    if metric == 'validation_accuracy':
        pbo.y_range=Range1d(0.984, 0.993)
    if metric == 'training_loss':
        pbo.y_range=Range1d(0, 0.15)
        
    pbo.xaxis.axis_label = 'Training epoch'
    pbo.yaxis.axis_label = metric
    output_file('plots/' + metric+".html", title=metric)
    show(pbo)

    x = list(max_acc.keys())
    pbo2 = figure(x_range=x, plot_width=1400, plot_height=500)
    pbo2.vbar(x, top=list(max_acc.values()), width=0.5)
    pbo2.y_range=Range1d(0.987, 0.993)
    pbo2.xaxis.major_label_orientation = 3.14/2
    pbo2.yaxis.axis_label = metric
    output_file('plots/' + metric+"_bar.html", title=metric)
    show(pbo2)
    

def hoyer_measure(x):
    r"""
        Hoyer's measure of sparsity
    """
    tmp1 = np.linalg.norm(x, 1)/(np.linalg.norm(x, 2))
    nel = x.size
    tmp2 = (np.sqrt(nel) - tmp1)/(np.sqrt(nel) - 1)
    return tmp2


def plot_weight_distribution():
    hoyer = {}
    hoyer_after_mean_subtraction = {}
    for reg_type in ['SO', 'BO', 'DO']:
        archs = get_experiment_configs(reg_type)
        for arch in archs:
            path = 'net_training/trained_data/'
            for k,v in arch.items():
                path += str(k) + '_' + str(v) + '_'
            if reg_type == 'DO':
                label = arch['name'][7:] + '_' + str(arch['drop_rate'])
            else:
                label = arch['name'][7:] + '_' + str(arch['drop_rate']) + '_' + str(arch['bo_norm'])
            net = LeNet5(arch)
            net.load_state_dict(torch.load(path[:-1]+'.model'))
            
            hoyer_per_layer = []
            hoyer_womean_per_layer = []
            mean_per_layer = []
            for l in ['conv1', 'conv2', 'fc1', 'fc2', 'fc3']:
                d = getattr(net, l).weight.data.cpu().numpy().flatten()
                hoyer_per_layer.append(hoyer_measure(d))
                m = np.mean(d)
                mean_per_layer.append(m)
                hoyer_womean_per_layer.append(hoyer_measure(d-m))
            hoyer[label] = hoyer_per_layer
            hoyer_after_mean_subtraction[label] = hoyer_womean_per_layer
    
    colors = ["#c9d9d3", "#718dbf", "#e84d60", "#7555bf", "#fffd60"]
    x = list(hoyer.keys())
    h = np.array(list(hoyer.values()))
    layers = ['conv1', 'conv2', 'fc1', 'fc2', 'fc3']
    data = {'x' : x,
        'conv1'   : h[:,0],
        'conv2'   : h[:,1],
        'fc1'     : h[:,2],
        'fc2'     : h[:,3],
        'fc3'     : h[:,4]}

    pbo2 = figure(x_range=x, plot_width=1400, plot_height=500)
    pbo2.vbar_stack(layers, x='x', width=0.8, color=colors, source=data,
             legend_label=layers)
    pbo2.xaxis.major_label_orientation = 3.14/2
    pbo2.yaxis.axis_label = 'Hoyer Measure'
    pbo2.y_range.start = 0
    pbo2.x_range.range_padding = 0.1
    pbo2.xgrid.grid_line_color = None
    pbo2.outline_line_color = None
    output_file('plots/' + "hoyer_measure_stacked.html", title='Hoyer Measure')
    show(pbo2)
    
    # this creates [ ("Apples", "2015"), ("Apples", "2016"), ("Apples", "2017"), ("Pears", "2015), ... ]
    xs = [ (r, l) for r in x for l in layers ]
    counts = sum(zip(data['conv1'], data['conv2'], data['fc1'], data['fc2'], data['fc3']), ()) # like an hstack
    
    source = ColumnDataSource(data=dict(x=xs, counts=counts))
    
    p = figure(x_range=FactorRange(*xs), plot_width=1400, plot_height=500)
    
    p.vbar(x='x', top='counts', width=0.7, source=source, line_color="white",

       # use the palette to colormap based on the the x[1:2] values
       fill_color=factor_cmap('x', palette=RdYlGn5, factors=layers, start=1, end=2))
    
    p.x_range.range_padding = 0.1
    p.xgrid.grid_line_color = None
    p.xaxis.major_label_orientation =3.14/2
    p.xaxis.group_label_orientation =3.14/2
    output_file('plots/' + "hoyer_measure_grouped.html", title='Hoyer Measure')    
    show(p)
    
    
def plot_sparsity():
    dash = cycle(['solid', 'dashed', 'dotted', 'dotdash', 'dashdot'])
    color = cycle(['red','blue','green','black','orange', 'cyan', 'purple','magenta','indigo'])
    path = 'net_training/trained_data/name_LeNet5_BO_All_drop_rate_0.3_bo_norm_2.5_validation_hmeasure'
    pbo = figure(plot_width=1400, plot_height=500)
    data = np.genfromtxt(path, delimiter=',')
    for h in range(4):
        c = pbo.line(range(len(data[:,h])),data[:,h], line_width=5, line_dash= next(dash), color=next(color), alpha=0.8, legend_label=str(h))
    pbo.y_range=Range1d(0.2, 0.65)
    pbo.legend.click_policy="hide"
    pbo.xaxis.axis_label = path
    show(pbo)

plot_accuracies()
# plot_sparsity()
# plot_weight_distribution()


