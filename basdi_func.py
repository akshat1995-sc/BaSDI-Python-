import numpy as np 
import cv2 as cv


def frm_to_img(data,h,w):

	img=np.zeros(shape=(h*w,))
	idx=np.array(data[:,0]+np.float(h)*data[:,1],dtype="int32")
	img[idx]=1
	img=img.reshape(h,w)
	return(img.T)

def gen_theta(data,h,w):
	data=np.concatenate(data[:])
	idx=np.array(data[:,0]+np.float(h)*data[:,1],dtype="int32")
	temp_hist=np.histogram(idx,range((h*w)+1))
	theta=temp_hist[0].reshape(h,w)
	return(theta.T)


def rmv_border(data,h,w,brder_size):
	"""Remove borders from frame data"""
	new_data=np.copy(data)
	for i in range(data.shape[0]):
		temp=np.c_[data[i]>=[brder_size,brder_size],data[i]<[h-brder_size,w-brder_size]]
		temp=temp[:,0] & temp[:,1] & temp[:,2] & temp[:,3]
		new_data[i]=data[i][temp]
	return (new_data)


def conv2d(x,y):
    y=y[::-1,::-1]
    temp_out=cv.filter2D(x,-1,y,borderType=0)
    H = np.floor(np.array(y.shape)/2).astype(np.int)
    outdims = (np.array(x.shape) - np.array(y.shape)) + 1
    temp_out=temp_out[H[0]:H[0]+outdims[0], H[1]:H[1]+outdims[1]]
    return(temp_out)

def psf_blur(theta,FHMW):
    s=FHMW/2.355
    theta=theta[::-1,::-1].astype('float')
    d=np.int(np.floor(FHMW*3))
    if d<5:
        d=5
    temp=cv.getGaussianKernel(d,s)
    kern=np.matmul(temp,temp.T)
    theta=cv.filter2D(theta,-1,kern,borderType=0)
    return(theta[::-1,::-1])

def theta2_convert(theta,scale,resolution):
    smooth=resolution*np.exp(scale)
    new_theta=psf_blur(theta,smooth)
    return(new_theta)



def likelihood(frms,theta,brder_size):
    """Evaluate P(o,0|x,y) for all support of x and y in [-30,30]"""
    if(theta.shape[0]<=brder_size+1 | theta.shape[1]<=brder_size+1):
        print("The size of the image does not comply with the density provided")
        return()
    [h,w]=theta.shape
    out=np.zeros(shape=(brder_size*2+1,brder_size*2+1,frms.shape[0]))
    default_prob=(brder_size*2+1)**(-2)
    logtheta=np.array(np.log(theta+1))			#Such that log(0) doesn't occur
    for i in range(frms.shape[0]):
		# For each frame
        if (frms[i].shape[0]==0):
            out[:,:,i]=default_prob
        else:
            img_data=frm_to_img(frms[i]-brder_size,h-2*brder_size,w-2*brder_size)
		# Shift (for certain drift) and multiply
            out_temp = conv2d(logtheta[::-1,::-1],img_data)
            out_temp= out_temp-(out_temp.max())
            out_temp=np.exp(out_temp[::-1,::-1])
            out[:,:,i]=out_temp/out_temp.sum()
    return(out)

def marginalize(likelihood):
    """Evaluate P(d|o,0)"""    
    [h,w,n_frms]=likelihood.shape;eps=0.001/500/500
    p=0.2;temp=cv.getGaussianKernel(3,np.sqrt(p))
    T=np.matmul(temp,temp.T)
    a=np.empty((h,w,n_frms))
    a[:,:,0]=likelihood[:,:,0]
    b=np.empty((h,w,n_frms))
    b[:,:,-1]=np.zeros((h,w))+1
    for i in range(1,n_frms):
        e_ia=likelihood[:,:,i]
        e_ib=likelihood[:,:,n_frms-i-1]
        a_temp=(cv.filter2D(a[::-1,::-1,i-1],-1,T,borderType=0)[::-1,::-1]+eps*np.sum(a[:,:,i-1]))*e_ia				#Add creep calculation
        t=b[:,:,n_frms-i]*e_ib
        b_temp=(cv.filter2D(t[::-1,::-1],-1,T,borderType=0)[::-1,::-1]+eps*np.sum(t))		#Add creep calculation
        a[:,:,i]=a_temp/np.max(a_temp)
        b[:,:,n_frms-i-1]=b_temp/np.max(b_temp)
    out=np.multiply(a,b)
    for i in range(n_frms):
        out[:,:,i]=out[:,:,i]/np.sum(out[:,:,i])
    return(out)


def update_theta(frms,drift_distri,h,w):
    theta=np.zeros((h,w))
    [dh,dw,n_frms]=drift_distri.shape
    max_sft=int((dh-1)/2)

    [x,y]=np.meshgrid(range(-max_sft,max_sft+1),range(-max_sft,max_sft+1))
    gn=drift_distri[:,:,0]/np.sum(drift_distri[:,:,0])
    cx=np.sum(np.multiply(x, gn))
    cy=np.sum(np.multiply(y, gn))
    print(cx)
    for i in range(frms.shape[0]):
        dk=drift_distri[:,:,i];img_data=frm_to_img(np.array(frms[i],dtype="int32"),h,w)
        theta=theta+cv.filter2D(img_data[::-1,::-1],-1,dk,borderType=0)[::-1,::-1]
    theta=np.roll(np.roll(theta,-1*np.round(cy).astype('int'),axis=0),-1*np.round(cx).astype('int'),axis=1)
    return(theta)

def drift_eval(drift_distri):
	[h,w,nfrms]=drift_distri.shape
	max_sft=int((h-1)/2)
	cx=np.zeros((nfrms,))
	cy=np.zeros((nfrms,))
	std_dev=np.zeros((nfrms,))
	[x,y]=np.meshgrid(range(-max_sft,max_sft+1),range(-max_sft,max_sft+1))
	for i in range(nfrms):
		gn=drift_distri[:,:,i]
		gn=gn/np.sum(gn)

		cx[i]=np.sum(x*gn)
		cy[i]=np.sum(y*gn)
		std_dev[i]=((np.sum((x**2)*gn)-cx[i])**2+(np.sum((y**2)*gn)-cy[i])**2)**(0.5)
	dout=np.array([-cy,-cx]).T
	return(dout,std_dev)

def test_convg(d_cur,d_prev):
	eps=0.001
	dt=np.std(d_cur-d_prev,axis=0)
	c=dt<np.std(d_cur,axis=0)*eps
	return(c)


