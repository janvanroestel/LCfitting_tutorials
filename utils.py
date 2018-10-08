import numpy as np
import ellc
import matplotlib.pyplot as plt

def plot(pars,data,model,rescale=False,shift=False):
    
    # generate the model
    m = model(data[:,0],pars)

    # rescale
    if shift:
        time = data[:,0]-pars[1]
    else:
        time = data[:,0]

    if rescale:
        m = m/pars[0]
        data[:,1] = data[:,1]/pars[0]
        data[:,2] = data[:,2]/pars[0]

    # 
    sigma = data[:,2]


    # Three subplots sharing both x/y axes
    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(time,m,'r-',zorder=4)
    ax1.errorbar(time,data[:,1],data[:,2],fmt='k.')
    ax1.errorbar(time,data[:,1],sigma,marker=None,color='0.8',
        lw=0,elinewidth=0.5,zorder=0)
    ax1.set_ylabel('Flux')


    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

    # show residual
    ax2.errorbar(time,data[:,1]-m,data[:,2],fmt='k.')
    ax2.errorbar(time,data[:,1]-m,sigma,marker=None,color='0.8',
        lw=0,elinewidth=0.5,zorder=0)
    ax2.set_ylabel('res. Flux')
    if shift:
        ax2.set_xlabel('time-t0 [JD]')
    else:
        ax2.set_xlabel('time [JD]')

    plt.show()

def make_initial_simplex(x0,scales):
    N = np.size(x0)
    initial_simplex = np.outer(x0,np.ones([N+1])).T
    initial_simplex[1:,:] += np.identity(np.size(x0))*10**scales
    return initial_simplex



def plot_chain(flatchain,nwalkers,save=False):

    N,npars = np.shape(flatchain)
    niter = N/nwalkers

    # make figure
    f, axes = plt.subplots(npars, sharex=True,figsize=(5,npars*2))
    f.subplots_adjust(hspace=0)

    for n in range(npars):
        ax = axes[n]
        walkers = flatchain[:,n].reshape(nwalkers,niter).T
        ax.plot(walkers,'k-',alpha=0.05)
        ax.plot(np.average(walkers,axis=1),'r-')
        ax.plot(np.percentile(walkers,q=50+34.1,axis=1),'r-',lw=0.5,alpha=0.5)
        ax.plot(np.percentile(walkers,q=50-34.1,axis=1),'r-',lw=0.5,alpha=0.5)
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

    plt.xlabel('iter')

    if save:
        plt.savefig('walkers.png',dpi=72*4)

    plt.show()
