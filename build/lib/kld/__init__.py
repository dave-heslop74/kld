# compile using: python3 setup.py sdist bdist_wheel
import numpy as np
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, Layout

def log_sinh(k):
    if k>700:
        s=k-np.log(2.0)
    else:
        s=np.log(np.sinh(k))
    return s

def KLD(MUp,Kp,MUq,Kq):
    
    term1 = np.log(Kp)+log_sinh(Kq)-np.log(Kq)-log_sinh(Kp)
    term2 = 1.0 / np.tanh(Kp) - 1.0 / Kp
    term3 = np.dot(MUp,(Kq*MUq - Kp*MUp).T)
    
    return term1-term2*term3


def calculator(IA,DA,KA,RA,IB,DB,KB,RB):

    muA = ID2XYZ(IA,DA)
    muB = ID2XYZ(IB,DB)
    
    K = KLD(muA,KA*RA,muB,KB*RB)
    output = widgets.HTML(value='<h4>Kullback Leibler Divergence = {0:.3f}</h4>'.format(K))
    
    Rtitle = widgets.HTML(value='<h3>Dissimilarity of paleomagnetic poles:</h3>')

    results=widgets.VBox((Rtitle,output))
    display(results)

def open_console(*args):
    
    style = {'description_width': 'initial'} #general style settings
    layout={'width': '220px'}
    
    spacer = widgets.HTML(value='<font color="white">This is some text!</font>')

    Atitle = widgets.HTML(value='<h4>Reference # compile using: python3 setup.py sdist bdist_wheel
import numpy as np
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, Layout

def log_sinh(k):
    if k>700:
        s=k-np.log(2.0)
    else:
        s=np.log(np.sinh(k))
    return s

def KLD0(MUp,Kp,MUq,Kq):
    
    term1 = np.log(Kp)+log_sinh(Kq)-np.log(Kq)-log_sinh(Kp)
    term2 = 1.0 / np.tanh(Kp) - 1.0 / Kp
    term3 = np.dot(MUp,(Kq*MUq - Kp*MUp).T)
    
    return term1-term2*term3


def calculator(IA,DA,KA,RA,IB,DB,KB,RB):

    muA = ID2XYZ(IA,DA)
    muB = ID2XYZ(IB,DB)
    
    K = np.squeeze(KLD0(muA,KA*RA,muB,KB*RB))
    output = widgets.HTML(value='<h4>Kullback Leibler Divergence = {0:.3f}</h4>'.format(K))
    
    Rtitle = widgets.HTML(value='<h3>Dissimilarity of paleomagnetic poles:</h3>')

    results=widgets.VBox((Rtitle,output))
    display(results)

def open_console(*args):
    
    style = {'description_width': 'initial'} #general style settings
    layout={'width': '220px'}
    
    spacer = widgets.HTML(value='<font color="white">This is some text!</font>')

    Atitle = widgets.HTML(value='<h4>Reference Pole p</h4>')
    IA=widgets.BoundedFloatText(value=0.0,min=-90.0,max=90.0,step=0.01,description='PLat$_p$ [-90$^\circ$: +90$^\circ$]:',style=style,layout=layout)
    DA=widgets.BoundedFloatText(value=0.0,min=0.0,max=360.0,step=0.01,description='Plon$_p$ [0$^\circ$: 360$^\circ$]:',style=style,layout=layout)
    KA=widgets.BoundedFloatText(value=0.01,min=0.0,max=100000,step=0.01,description='$\kappa_p$ [>0]:',style=style,layout=layout)
    RA=widgets.BoundedFloatText(value=1,min=1,max=100000,step=0.01,description='$R_p$ [$\geq$1]:',style=style,layout=layout)

    Btitle = widgets.HTML(value='<h4>Pole q</h4>')
    IB=widgets.BoundedFloatText(value=0.0,min=-90.0,max=90.0,step=0.01,description='Plat$_q$ [-90$^\circ$: +90$^\circ$]:',style=style,layout=layout)
    DB=widgets.BoundedFloatText(value=0.0,min=0.0,max=360.0,step=0.01,description='Plon$_q$ [0$^\circ$: 360$^\circ$]:',style=style,layout=layout)
    KB=widgets.BoundedFloatText(value=0.01,min=0.0,max=100000,step=0.01,description='$\kappa_q$ [>0]:',style=style,layout=layout)
    RB=widgets.BoundedFloatText(value=1,min=1.0,max=100000,step=0.01,description='$R_q$ [$\geq$1]:',style=style,layout=layout)
    
    uA = widgets.VBox((Atitle,IA, DA, KA, RA),layout=Layout(overflow_y='initial',height='180px'))
    uB = widgets.VBox((Btitle,IB, DB, KB, RB),layout=Layout(overflow_y='initial',height='180px'))
    uAB = widgets.HBox((uA,spacer,uB),layout=Layout(overflow_y='initial',height='180px'))
    ui = widgets.VBox([uAB],layout=Layout(overflow_y='initial',height='180px')) 

    out = widgets.interactive_output(calculator, {'IA': IA, 'DA': DA, 'KA': KA, 'RA': RA, 'IB': IB, 'DB': DB, 'KB': KB, 'RB': RB})
    display(ui,out)

def ID2XYZ(I,D):
    
    I = np.deg2rad(I)
    D = np.deg2rad(D)
    
    XYZ=np.column_stack((np.cos(D)*np.cos(I),np.sin(D)*np.cos(I),np.sin(I)))
    
    return XYZ

Pole p</h4>')
    IA=widgets.BoundedFloatText(value=0.0,min=-90.0,max=90.0,step=0.01,description='PLat$_p$ [-90$^\circ$: +90$^\circ$]:',style=style,layout=layout)
    DA=widgets.BoundedFloatText(value=0.0,min=0.0,max=360.0,step=0.01,description='Plon$_p$ [0$^\circ$: 360$^\circ$]:',style=style,layout=layout)
    KA=widgets.BoundedFloatText(value=0.01,min=0.0,max=100000,step=0.01,description='$\kappa_p$ [>0]:',style=style,layout=layout)
    RA=widgets.BoundedFloatText(value=1,min=1,max=100000,step=0.01,description='$R_p$ [$\geq$1]:',style=style,layout=layout)

    Btitle = widgets.HTML(value='<h4>Pole q</h4>')
    IB=widgets.BoundedFloatText(value=0.0,min=-90.0,max=90.0,step=0.01,description='Plat$_q$ [-90$^\circ$: +90$^\circ$]:',style=style,layout=layout)
    DB=widgets.BoundedFloatText(value=0.0,min=0.0,max=360.0,step=0.01,description='Plon$_q$ [0$^\circ$: 360$^\circ$]:',style=style,layout=layout)
    KB=widgets.BoundedFloatText(value=0.01,min=0.0,max=100000,step=0.01,description='$\kappa_q$ [>0]:',style=style,layout=layout)
    RB=widgets.BoundedFloatText(value=1,min=1.0,max=100000,step=0.01,description='$R_q$ [$\geq$1]:',style=style,layout=layout)
    
    uA = widgets.VBox((Atitle,IA, DA, KA, RA),layout=Layout(overflow_y='initial',height='180px'))
    uB = widgets.VBox((Btitle,IB, DB, KB, RB),layout=Layout(overflow_y='initial',height='180px'))
    uAB = widgets.HBox((uA,spacer,uB),layout=Layout(overflow_y='initial',height='180px'))
    ui = widgets.VBox([uAB],layout=Layout(overflow_y='initial',height='180px')) 

    out = widgets.interactive_output(calculator, {'IA': IA, 'DA': DA, 'KA': KA, 'RA': RA, 'IB': IB, 'DB': DB, 'KB': KB, 'RB': RB})
    display(ui,out)

def ID2XYZ(I,D):
    
    I = np.deg2rad(I)
    D = np.deg2rad(D)
    
    XYZ=np.column_stack((np.cos(D)*np.cos(I),np.sin(D)*np.cos(I),np.sin(I)))
    
    return XYZ

