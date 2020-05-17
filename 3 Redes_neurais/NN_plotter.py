# -*- coding: utf-8 -*-
"""
Last editted on Thu Jul 25 09:32:57 2019

@author: Matheus Di Vaio
"""



from bokeh.models import ColumnDataSource, Range1d, HoverTool, Toolbar, ToolbarBox
from bokeh.models.tools import (
    HoverTool,
    WheelZoomTool,
    PanTool,
    CrosshairTool,
    LassoSelectTool,
)
from bokeh.layouts import layout


from bokeh.models import ColumnDataSource, Range1d, HoverTool
from bokeh.layouts import row, column, gridplot, layout

from bokeh.models import Button, ColumnDataSource, Range1d, Toolbar, ToolbarBox
from bokeh.models.tools import HoverTool, WheelZoomTool, PanTool, CrosshairTool

from bokeh.layouts import layout
from bokeh.plotting import curdoc, figure, save

from bokeh.models import Button, ColumnDataSource, Range1d, Toolbar, ToolbarBox, LinearAxis

from bokeh.layouts import layout
from bokeh.plotting import curdoc, figure

from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, ColorBar

from bokeh.models import Span, Label

from bokeh.io import export_svgs, export_png

# from bokeh.palettes import Spectral6
from bokeh.transform import linear_cmap
import bokeh.palettes as bp

import numpy as np
from copy import deepcopy

xwheel_zoom = WheelZoomTool(dimensions="width")
pan_tool = PanTool()
hover = HoverTool()
crosshair = CrosshairTool()

#    tools = (xwheel_zoom, pan_tool, hover, crosshair)

toolbar = Toolbar(
    tools=[xwheel_zoom, pan_tool, hover, crosshair],
    active_inspect=[crosshair],
    # active_drag =                         # here you can assign the defaults
    # active_scroll =                       # wheel_zoom sometimes is not working if it is set here
    # active_tap
)

toolbar_box = ToolbarBox(toolbar=toolbar, toolbar_location="above")




def generate_figures(
    metrics,
    best_epoch,
    y_train,
#    y_train_pred,
    y_test,
#    y_test_pred,
    X_train,
    X_test,
    seq_len,
    nome_rodada,
    nome_title,
    save_figs=False,
    show_figs=True
):

    
    
    fig_training_error = plot_training_error(metrics, best_epoch, nome_rodada)
    


    y = np.vstack(
        [
            X_train[:, 0],
            X_train[:, 1],
            X_train[:, 2],
            y_train[:len(X_train)].reshape(len(X_train)),
#            X_train[:, 5],
        ]
    ).T
    fig_train_pred_stc = plot_generic(
        y=y,
        color=["blue", "orangered", "forestgreen", "black", "red"],
        
        legend=["Mean SS", "Mean SS-2std", "Mean SS+2std", "Exp. SS", "std"],
        xlabel="Sample",
        ylabel="SS",
        yrange=[0, 3.5],
        title="Training dataset - Stochastic Prediction",
        plot_type=["line", "scatter", "scatter", "line", "scatter"],
        ALPHA=[0.9, 0.8, 0.8, 1, 1],
    )



    y = np.vstack(
        [
            X_test[:, 0],
            X_test[:, 1],
            X_test[:, 2], 
            y_test[:len(X_test)].reshape(len(X_test)),
#            X_test[:, 5],  
        ]
    ).T

    fig_test_pred_stc = plot_generic(
        y=y,
        color=["blue", "orangered", "forestgreen", "black", "red"],
        legend=["Mean SS", "Mean SS-2std", "Mean SS+2std", "Exp. SS", "std"],
        xlabel="Sample",
        ylabel="SS",
        yrange=[0, 3.5],
        title="Testing dataset - Stochastic Prediction",
        plot_type=["line", "scatter", "scatter", "line", "scatter"],
        ALPHA=[0.9, 0.8, 0.8, 1, 1],
    )
    
  
    
    
    
#    if save_figs is True:
#        fig2 = fig_train_pred_stc
#        output_file('Pred_train' + nome_rodada + ".html")
#        save(fig2)
#        
#        fig3 = fig_test_pred_stc
#        output_file('Pred_test' + nome_rodada + ".html")
#        save(fig3)
#        
##    if save_figs is True:
##        fig2 = fig_train_pred_stc
##        
##        fig2.output_backend = "svg"
##        export_svgs(fig2, filename='Pred_train' + nome_rodada + ".png")
##        
##        
##        
##        fig3 = fig_test_pred_stc
##        fig3.output_backend = "svg"
##        export_svgs(fig3, filename='Pred_test' + nome_rodada + ".png")
        
    
#    else:
    fig1 = fig_training_error
    fig2 = fig_train_pred_stc
    fig3 = fig_test_pred_stc

    
    
#    grid = [[fig2], [fig3]]
#    show(fig1)
#    plot_layoult(title='Pred STC '+nome_rodada, save_figs=save_figs, show_figs=show_figs, grid1=grid, grid2=None)


    return [[fig1], [fig2], [fig3]]
        
        
def plot_generic_2yaxis(
        
    x=None,
    y=None,
    color=None,
    legend=None,
    title="title",
    xlabel="xlabel",
    ylabel="ylabel",
    xrange=None,
    yrange=None,
    plot_type="line",
    ALPHA=None,
):

    if len(y.shape) == 1:
        num_plots = 1
        y = y.reshape(len(y), 1)
    else:
        num_plots = y.shape[1]

        if type(plot_type) == str:
            plot_type = [plot_type]

        if len(plot_type) < num_plots:
            for i in range(len(plot_type), num_plots):
                plot_type = np.hstack([plot_type, plot_type[i - 1]])

    if x is None:
        x = np.arange(y.shape[0])

    if ALPHA is None:
        ALPHA = np.ones(num_plots)
    if color is None:
        color = ["navy", "green"]

    BALL_SIZE = 2

    fig = figure(
        x_axis_label=xlabel,
        y_axis_label=ylabel,
        x_range=xrange,
        y_range=yrange,
        sizing_mode="stretch_both",
        toolbar_location='right',#"above",
        title=title,
    )
    
    fig.extra_y_ranges = {"std_axys": Range1d(start=-0, end=0.5)}
    fig.add_layout(LinearAxis(y_range_name="std_axys"), 'right')

    for i in range(num_plots):
        source = ColumnDataSource(data=dict(x=x, y=y[:, i]))
        if i == num_plots-1:
            fig.circle(
                "x",
                "y",
                legend=legend[i],
                size=BALL_SIZE,
                source=source,
                
                color=color[i],
                alpha=ALPHA[i],
                y_range_name='std_axys',
            )
            
        else:
            if plot_type[i] == "scatter":
                fig.circle(
                    "x",
                    "y",
                    legend=legend[i],
                    size=BALL_SIZE,
                    source=source,
                    color=color[i],
                    alpha=ALPHA[i],
                )
    
            if plot_type[i] == "line":
                fig.line(
                    "x",
                    "y",
                    legend=legend[i],
                    line_width=2,
                    source=source,
                    color=color[i],
                    alpha=ALPHA[i],
                )

    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"
    
    return fig



def plot_generic(
        
    x=None,
    y=None,
    color=None,
    legend=None,
    title="title",
    xlabel="xlabel",
    ylabel="ylabel",
    xrange=None,
    yrange=None,
    plot_type="line",
    ALPHA=None,
):

    if len(y.shape) == 1:
        num_plots = 1
        y = y.reshape(len(y), 1)
    else:
        num_plots = y.shape[1]

        if type(plot_type) == str:
            plot_type = [plot_type]

        if len(plot_type) < num_plots:
            for i in range(len(plot_type), num_plots):
                plot_type = np.hstack([plot_type, plot_type[i - 1]])

    if x is None:
        x = np.arange(y.shape[0])

    if ALPHA is None:
        ALPHA = np.ones(num_plots)
    if color is None:
        color = ["navy", "green"]

    BALL_SIZE = 1.5

    fig = figure(
        x_axis_label=xlabel,
        y_axis_label=ylabel,
        x_range=xrange,
        y_range=yrange,
#        sizing_mode="stretch_both",
#        toolbar_location='none', #'right',"above",
        title=title,
        plot_width=950, plot_height=350,
    )

    for i in range(num_plots):
        source = ColumnDataSource(data=dict(x=x, y=y[:, i]))
        if plot_type[i] == "scatter":
            fig.circle(
                "x",
                "y",
                legend=legend[i],
                size=BALL_SIZE,
                source=source,
                color=color[i],
                alpha=ALPHA[i],
            )

        if plot_type[i] == "line":
            fig.line(
                "x",
                "y",
                legend=legend[i],
                line_width=1,
                source=source,
                color=color[i],
                alpha=ALPHA[i],
            )

    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"
    
    return fig



def plot_training_error(y, best_epoch, nome_title):
    
    
    
    
    if best_epoch != np.argmin(y[1:,1]):
        lowest_loss = y[best_epoch, 1]
    else:   
        best_epoch = np.argmin(y[1:,1])
        lowest_loss = np.min(y[1:,1])
    
    
    
    legend=["Train error", "Test error"]
    color=["blue", "red"]

    x = np.arange(y.shape[0])

    fig = figure(
        x_axis_label="Epoch",
        y_axis_label="Error",
#        sizing_mode="stretch_both",
#        sizing_mode="fixed",
        plot_width=800, plot_height=400,
        title="Error vs epoch" + nome_title,
        y_axis_type="log",
        toolbar_location=None,
        
    )
    
    for i in range(y.shape[1]):
        source = ColumnDataSource(data=dict(x=x, y=y[:, i]))

        fig.line("x", "y",
            legend=legend[i],
            line_width=2,
            source=source,
            color=color[i],
            alpha=1,
        )

#    fig.circle(
#               x=best_epoch+1,
#               y=lowest_loss,
#        legend='Best epoch',
#        size=30,
#        line_width=3,
#        line_color='green',
#        fill_color=None,
##            color='red',
#        alpha=1,
#    )

    v_line = Span(location=best_epoch+1,
                  dimension='height', 
                  line_color='green',
                  line_width=2,
                  )
    fig.add_layout(v_line)
    
#    fig.add_layout(
#            Label(
#                x=best_epoch-2,
##                x=0,
#                y=lowest_loss + 0.001,
#                angle=0,
#                text='Best Epoch: ' + str(best_epoch+1)+'  Test error: {0:.3f}'.format(lowest_loss, 5),
#                
#                text_font_style="bold",
#                text_font_size="12pt",
#                text_baseline="top",
#                text_align="right",
#                x_offset=20,
#                y_offset=200,
#            )
#        )
    
    
    fig.legend.location = "bottom_left"
    fig.legend.click_policy = "hide"
    

    
    return fig



def plot_layoult(title, grid1, save_figs=False, show_figs=True, grid2=None):

    if title is None:
        title='NOME DA FIGURA NAO FORNECIDO'
        
    xwheel_zoom = WheelZoomTool(dimensions="width")
    pan_tool = PanTool()
    hover = HoverTool()
    crosshair = CrosshairTool()

    toolbar = Toolbar(
        tools=[xwheel_zoom, pan_tool, hover, crosshair],
        active_inspect=[crosshair],
        # active_drag =                         # here you can assign the defaults
        # active_scroll =                       # wheel_zoom sometimes is not working if it is set here
        # active_tap
    )

    toolbar_box = ToolbarBox(toolbar=toolbar, toolbar_location="above")

    if grid2 is None:
        layout_2 = layout(children=[[
#                toolbar_box, 
                grid1
                ]], 
#                          sizing_mode="stretch_both",
#                          plot_width=400, plot_height=800,
                          )
    else:
        layout_2 = layout(children=[[
#                toolbar_box, 
                [[grid1], 
                [grid2]]
                ]],
#            sizing_mode="stretch_both",
#            plot_width=3000, plot_height=1000,
            )

    
    
    
    if save_figs is True:
        output_file(title + ".html")
        if show_figs is True:
            show(layout_2)
        else:
            save(layout_2)
    else:
        show(layout_2)
        






def plot_histogram2D(W_hist, B_hist, nome_rodada, nome_title, plotW = True, plotB = True, save_figs=False, show_figs=True):


    grid_W_hist = generate_hist2D_figures(W_hist, bias = False, TITLE='Weights Histogram'+nome_title)
    grid_B_hist = generate_hist2D_figures(B_hist, bias = True, TITLE='Bias Histogram'+nome_title)


#    plot_layoult(grid1=grid_W_hist, grid2=grid_B_hist[:-1], title='Histogram'+ nome_rodada, save_figs=save_figs, show_figs=show_figs)
    plot_layoult(title='Histogram'+ nome_rodada, grid1=grid_W_hist, save_figs=save_figs, show_figs=show_figs, grid2=grid_B_hist[:-1])



def generate_hist2D_figures(Matrix, bias = False, metrics = None, TITLE = None):
    H = Matrix[0]
    xedges = Matrix[1]
    yedges = Matrix[2]

    grid = []

    for i in range(len(H)):
        
        title = None
        x_label = 'Epoch - Layer: ' + str(i)
        
        if bias is False:
            if i == 0:
                title = TITLE
                x_label = 'Epoch - Input Layer'
                
        p = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")],
                     x_range = [np.min(xedges[i]), np.max(xedges[i])],
                     y_range = [np.min(yedges[i]), np.max(yedges[i])],
                     x_axis_label=x_label,
                     sizing_mode="stretch_both",
                     title = title,
                     title_location = 'above'
        )
        
        d = H[i].T
        d2 = np.log(d)
        d3 = d2
        
        from numpy import inf
        d3[d3 == -inf] = 0
        d=d3
        p.image(image=[d], 
                x=np.min(xedges[i]), 
                y=np.min(yedges[i]), 
                dw=np.max(xedges[i]) - np.min(xedges[i]), 
                dh=np.max(yedges[i]) - np.min(yedges[i]), 
                palette="RdYlBu11")
        
        grid.append(p)
        
    return grid








def plot_histogram(X_train, y_train, y_train_eq, X_test, y_test, bins, normalizador, nome_rodada, save_figs, show_figs):
    
    hist_y_train,    edges_y_train    = np.histogram(y_train*normalizador,    density=False, bins=bins, range=(0,normalizador))
    hist_y_train_eq, edges_y_train_eq = np.histogram(y_train_eq*normalizador, density=False, bins=bins, range=(0,normalizador))
    hist_y_test,     edges_y_test     = np.histogram(y_test*normalizador,     density=False, bins=bins, range=(0,normalizador))

    X = [X_train, X_test]
    
    Y = [y_train*normalizador, y_test*normalizador]
    
    SS    = [[], []]
    SS_std= [[], []]
    hist  = [[], []]
    edges = [[], []]
    error = [[], []]
    
    for i in range(0, len(X)):
        count = 0
        x = X[i]
        y = Y[i]
        
        for j in range(x.shape[0]):
            if y[j] > x[j,1] or y[j] < x[j,2]:
                
                SS[i].append(y[j]) 
                
                if x[j,5] <= 0.01:
                    count += 1
                    pass
                else:
                    SS_std[i].append(abs(y[j] - x[j,0]) / x[j,5])
                
#                print(y[j], x[j,1], x[j,2])
                
        print(len(x), len(SS[i]), len(SS_std[i]), count)
        
        hist[i], edges[i] = np.histogram(SS[i], density=False, bins=bins, range=(0,normalizador))

        error[i] = len(SS[i])/len(x)
    
#    SS     = [[np.asarray(SS[0])],     [np.asarray(SS[1])]]
#    SS_std = [[np.asarray(SS_std[0][0])], [np.asarray(SS_std[1][0])]]
#     + '% Mean error std: {0:.0f}'.format(np.mean(SS_std[0]), 2)
    fig_ytrain    = generade_hist('Histogram SS train: Error:  {0:.0f}'.format(error[0]*100, 2) + '%',    
                                  hist_y_train,    edges_y_train, hist[0], edges[0], hist_y_train_eq, edges_y_train)

    fig_ytest     = generade_hist('Histogram SS test: Error:  {0:.0f}'.format(error[1]*100, 2) + '%',    
                                  hist_y_test,     edges_y_test,  hist[1], edges[1])
   
    fig_ytrain_PURE    = generade_hist('Histogram SS train: Error:  {0:.0f}'.format(error[0]*100, 2) + '%',    
                                  hist_y_train,    edges_y_train)
    
    fig_ytrain_std = generade_error_scatter('Histogram2D SS_std', np.asarray(SS[0]), np.asarray(SS_std[0]), normalizador)
    
    fig_ytest_std  = generade_error_scatter('Histogram2D SS_std', np.asarray(SS[1]), np.asarray(SS_std[1]), normalizador)
    
    
    
    

#    grid = [[fig_ytrain], [fig_ytest]]
#    
#    grid2 = [[fig_ytrain_std], [fig_ytest_std]]
    
#    grid = [[fig_ytrain , fig_ytest], [fig_ytrain_std, fig_ytest_std]]
    grid = [[fig_ytrain , fig_ytrain_std], [fig_ytest, fig_ytest_std]]
    
#    plot_layoult(title='Histogram Error ' + nome_rodada, grid1=grid, grid2=grid2, save_figs=save_figs, show_figs=True)
    
    
#    plot_layoult(title='Histogram Error ' + nome_rodada, grid1=grid, grid2=None, save_figs=save_figs, show_figs=True)


    return grid

    
def generade_hist(title, hist0, edges0, hist1=None, edges1=None, hist2=None, edges2=None):
    p = figure(title=title,
               plot_width=400, plot_height=200,
#               background_fill_color="#fafafa",
#               toolbar_location = 'none',
               )
    
        
    if (hist2 is not None) or (edges2 is not None):
        p.quad(top=hist2, bottom=0, left=edges2[:-1], right=edges2[1:],
           fill_color="navy", line_color="white", alpha=0.5)

    p.quad(top=hist0, bottom=0, left=edges0[:-1], right=edges0[1:],
           fill_color="navy", line_color="white", alpha=1)
    
    if (hist1 is not None) or (edges1 is not None):
        p.quad(top=hist1, bottom=0, left=edges1[:-1], right=edges1[1:],
           fill_color="red", line_color="white", alpha=0.7)
        
        
    p.y_range.start = 0
    p.legend.location = "center_right"
    p.legend.background_fill_color = "white" #"#fefefe"
    p.xaxis.axis_label = 'SS value'
    p.yaxis.axis_label = 'Number of samples'
#    p.grid.grid_line_color="white"
#    show(p)
    return p


def generade_error_scatter(title, ss, ss_std, normalizador):
    binx = 40
    biny = 40
    



    x = np.linspace(2, 8, binx)
    y = np.linspace(0, normalizador, biny)
    
    density = np.zeros([biny-1,binx-1])

    for j in range(binx-1):
        for k in range(biny-1):
            count = 0
            for l in range(len(ss_std)):
                
                if (ss_std[l] >= x[j]) and (ss_std[l] <= x[j+1]) and (ss[l] >= y[k]) and (ss[l] <= y[k+1]):
                    count += 1
            
       
            density[k,j] = count
            
    
    valor = 0        
    for i in range(len(ss_std)):
        if ss_std[i] >= 2 and ss_std[i] <= 3:
            valor += 1
    
    valor = valor/len(ss_std)
            
            
    H = density
    xedges = x
    yedges = y

            
    p = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")],
                 x_range = [np.min(xedges), np.max(xedges)],
                 y_range = [np.min(yedges), np.max(yedges)],
#                     x_axis_label=x_label,
#                 sizing_mode="stretch_both",
               plot_width=400, plot_height=200,
                 title = 'Errors between 2<std<3: {0: .0f}'.format(valor*100) + '%',
                 title_location = 'above',
#                 toolbar_location = 'none',
    )
    
    d = H
    d2 = np.log(d)
    d3 = d2
#        
    from numpy import inf
    d3[d3 == -inf] = 0
    d=d3
    
    p.image(image=[d], 
            x=np.min(xedges), 
            y=np.min(yedges), 
            
            
            
            dw=np.max(xedges) - np.min(xedges), 
            dh=np.max(yedges) - np.min(yedges), 
            palette="RdYlBu11")
       
    p.yaxis.axis_label = 'SS value'
    p.xaxis.axis_label = 'Distance in std' 
#    show(p)
    
    
    return p
        






def plot_clusters(dataY, clusters, nome_rodada, nome_title, save_figs=False, show_figs=True):
    
    train = clusters[0]
    test  = clusters[1]
    val   = clusters[2]
    
    train_coord = []
    test_coord = []
    val_coord = []
    for i in range(len(train)):
        if len(train[i]) != 0:            
            train_coord += [train[i][0], train[i][-1]]
    for i in range(len(test)): 
        if len(test[i]) != 0:
            test_coord += [test[i][0], test[i][-1]]
    for i in range(len(val)): 
        if len(val[i]) != 0: 
            val_coord += [val[i][0], val[i][-1]]
    
 
    
    y_coord = np.ones(len(train_coord))*3
    for i in range(len(train_coord)):
        if i % 2 == 0:
            y_coord[i] = 0      
    
    train_coord,
    test_coord,
    test_coord
    

        
    p = figure(x_axis_label='Data ',
               sizing_mode="stretch_both",
               title = 'Clusters' + nome_title,
               title_location = 'above'
        )
     
    for i in range(0, len(train_coord), 2):
        
        p.patch(x=[train_coord[i], train_coord[i+1], train_coord[i+1], train_coord[i]],
                y=[0, 0, 3.5, 3.5], color="green", fill_alpha=0.5)
        
    for i in range(0, len(test_coord), 2):
        p.patch(x=[test_coord[i], test_coord[i+1], test_coord[i+1], test_coord[i]], 
                y=[0, 0, 3.5, 3.5], color="red", fill_alpha=0.5)
   
    x = np.arange(len(dataY))
    p.line(x = x, y=dataY.reshape(-1,), color="black") 
    

    p.output_backend = "svg"
    export_svgs(p, filename=('Clusters'+ nome_rodada + "plot.svg"))
    
#    if save_figs is True:
#        output_file('Clusters'+ nome_rodada +'.html')
#        if show_figs is True:
#            show(p)
#        else:
#            save(p)
#    else:
#        show(p)




    
    
    
    
    