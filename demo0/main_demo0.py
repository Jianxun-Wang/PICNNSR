import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
from torch.utils.data import DataLoader
import time
import tikzplotlib
from sklearn.metrics import mean_squared_error as calMSE

from readVesselMesh import read_vessel_mesh,\
						   proj_solu_from_ofmesh_2_mymesh
from dataset import VaryGeoDataset
from pyMesh import visualize2D, setAxisLabel,to4DTensor
from model import USCNNSep
from readOF import convertOFMeshToImage_StructuredMesh



h=0.01
Itol=0
#Prepare data
mesh_hf,nx_hf,ny_hf=read_vessel_mesh('ns_eqn_parabolic/3200/C')
solu_hf1=convertOFMeshToImage_StructuredMesh(nx_hf,ny_hf,'ns_eqn_parabolic/3200/C',
	                                             ['ns_eqn_parabolic/3200/U',
	                                              'ns_eqn_parabolic/3200/p'],
	                                            [0,1,0,1],0.0,False)
solu_hf2=convertOFMeshToImage_StructuredMesh(nx_hf,ny_hf,'ns_eqn_parabolic/3200/C',
	                                             ['ns_eqn_parabolic/3200/T'],
	                                            [0,1,0,1],0.0,False)
solu_hf=np.concatenate([solu_hf1,solu_hf2[:,:,2:]],axis=2)
mesh_lf,nx_lf,ny_lf=read_vessel_mesh('Coarse/3200/C')
solu_lf=convertOFMeshToImage_StructuredMesh(nx_lf,ny_lf,'Coarse/3200/C',
	                                           ['Coarse/3200/U',
	                                            'Coarse/3200/p'], 
												[0,1,0,1],1,False)
[OFU_lf,OFV_lf,OFP_lf]=proj_solu_from_ofmesh_2_mymesh(solu_lf,mesh_lf)
relativeNoise=0#np.random.uniform(low=0.0, high=1.0, size=1)
inputRand=relativeNoise*np.random.rand(1, 2, 14, 9)
inputRand=torch.from_numpy(inputRand)
inputRand=inputRand.float().to('cuda')
u_lf=torch.from_numpy(OFU_lf).float().to('cuda').reshape([1,1,OFU_lf.shape[0],OFU_lf.shape[1]])
v_lf=torch.from_numpy(OFV_lf).float().to('cuda').reshape([1,1,OFV_lf.shape[0],OFV_lf.shape[1]])
p_lf=torch.from_numpy(OFP_lf).float().to('cuda').reshape([1,1,OFP_lf.shape[0],OFP_lf.shape[1]])
velofunc=lambda argc: 100*argc*(0.2-argc)
#pdb.set_trace()
infertruth=torch.from_numpy(velofunc(mesh_hf.x[0,:])).float().to('cuda')
infertruth=infertruth[1:-1].reshape([1,47])
solu_hf1=convertOFMeshToImage_StructuredMesh(nx_hf,ny_hf,'ns_eqn_parabolic/3200/C',
	                                             ['ns_eqn_parabolic/3200/U',
	                                              'ns_eqn_parabolic/3200/p'],
	                                            [0,1,0,1],0.0,False)
solu_hf2=convertOFMeshToImage_StructuredMesh(nx_hf,ny_hf,'ns_eqn_parabolic/3200/C',
	                                             ['ns_eqn_parabolic/3200/T'],
	                                            [0,1,0,1],0.0,False)
solu_hf=np.concatenate([solu_hf1,solu_hf2[:,:,2:]],axis=2)
HFMeshList=[mesh_hf]
train_set=VaryGeoDataset(HFMeshList)


#Set up paramter
batchSize=1
NvarInput=2
NvarOutput=1
nEpochs=100000
lr=0.001
Ns=1
nu=0.01
criterion = nn.MSELoss()
padSingleSide=1
udfpad=nn.ConstantPad2d([padSingleSide,padSingleSide,padSingleSide,padSingleSide],0)

#Set up model
model=USCNNSep(h,nx_hf,ny_hf,NvarInput,NvarOutput,'ortho').to('cuda')
model=model.to('cuda')
optimizer = optim.Adam(model.parameters(),lr=lr)
training_data_loader=DataLoader(dataset=train_set,
	                            batch_size=batchSize)

#Project OFSolution 2 MyMesh
[OFU_sb,OFV_sb,OFP_sb,OFT_sb]=proj_solu_from_ofmesh_2_mymesh(solu_hf,mesh_hf)
#[BICUBICU,BICUBICV,_]=proj_solu_from_ofmesh_2_mymesh(solu_lf,mesh_hf)

T_hardimpo=torch.from_numpy(OFT_sb).reshape([1,1,OFT_sb.shape[0],OFT_sb.shape[1]]).float().to('cuda')
OFV_sb_cuda=torch.from_numpy(OFV_sb).reshape([1,1,OFV_sb.shape[0],OFV_sb.shape[1]]).float().to('cuda')

#Specify OBS
from random import sample
nobs=200
idxpool=[i for i in range(77*49) if i>147]
#idxsample=sample(idxpool,nobs)
idxsample=[i for i in idxpool if (i in range(49*10+2,49*11-2)) or (i in range(49*30+2,49*31-2)) or (i in range(49*50+2,49*51-2)) or (i in range(49*65+2,49*66-2))]
idx1=[i//49 for i in idxsample] #77
idx2=[i%49 for i in idxsample] #49
idx1=np.asarray(idx1)
np.savetxt('idx1.txt',idx1)
idx2=np.asarray(idx2)
np.savetxt('idx2.txt',idx2)

#Define the geometry transformation
def dfdx(f,dydeta,dydxi,Jinv):
	dfdxi_internal=(-f[:,:,:,4:]+8*f[:,:,:,3:-1]-8*f[:,:,:,1:-3]+f[:,:,:,0:-4])/12/h	
	dfdxi_left=(-11*f[:,:,:,0:-3]+18*f[:,:,:,1:-2]-9*f[:,:,:,2:-1]+2*f[:,:,:,3:])/6/h
	dfdxi_right=(11*f[:,:,:,3:]-18*f[:,:,:,2:-1]+9*f[:,:,:,1:-2]-2*f[:,:,:,0:-3])/6/h
	dfdxi=torch.cat((dfdxi_left[:,:,:,0:2],dfdxi_internal,dfdxi_right[:,:,:,-2:]),3)
	
	dfdeta_internal=(-f[:,:,4:,:]+8*f[:,:,3:-1,:]-8*f[:,:,1:-3,:]+f[:,:,0:-4,:])/12/h	
	dfdeta_low=(-11*f[:,:,0:-3,:]+18*f[:,:,1:-2,:]-9*f[:,:,2:-1,:]+2*f[:,:,3:,:])/6/h
	dfdeta_up=(11*f[:,:,3:,:]-18*f[:,:,2:-1,:]+9*f[:,:,1:-2,:]-2*f[:,:,0:-3,:])/6/h
	dfdeta=torch.cat((dfdeta_low[:,:,0:2,:],dfdeta_internal,dfdeta_up[:,:,-2:,:]),2)
	dfdx=Jinv*(dfdxi*dydeta-dfdeta*dydxi)
	return dfdx

def dfdy(f,dxdxi,dxdeta,Jinv):
	dfdxi_internal=(-f[:,:,:,4:]+8*f[:,:,:,3:-1]-8*f[:,:,:,1:-3]+f[:,:,:,0:-4])/12/h	
	dfdxi_left=(-11*f[:,:,:,0:-3]+18*f[:,:,:,1:-2]-9*f[:,:,:,2:-1]+2*f[:,:,:,3:])/6/h
	dfdxi_right=(11*f[:,:,:,3:]-18*f[:,:,:,2:-1]+9*f[:,:,:,1:-2]-2*f[:,:,:,0:-3])/6/h
	dfdxi=torch.cat((dfdxi_left[:,:,:,0:2],dfdxi_internal,dfdxi_right[:,:,:,-2:]),3)
	
	dfdeta_internal=(-f[:,:,4:,:]+8*f[:,:,3:-1,:]-8*f[:,:,1:-3,:]+f[:,:,0:-4,:])/12/h	
	dfdeta_low=(-11*f[:,:,0:-3,:]+18*f[:,:,1:-2,:]-9*f[:,:,2:-1,:]+2*f[:,:,3:,:])/6/h
	dfdeta_up=(11*f[:,:,3:,:]-18*f[:,:,2:-1,:]+9*f[:,:,1:-2,:]-2*f[:,:,0:-3,:])/6/h
	dfdeta=torch.cat((dfdeta_low[:,:,0:2,:],dfdeta_internal,dfdeta_up[:,:,-2:,:]),2)
	dfdy=Jinv*(dfdeta*dxdxi-dfdxi*dxdeta)
	return dfdy

# Define the model to train
def train(epoch):
	startTime=time.time()
	xRes=0
	yRes=0
	mRes=0
	TRes=0
	eU=0
	eV=0
	eP=0
	for iteration, batch in enumerate(training_data_loader):
		[_,cord,_,_,_,Jinv,dxdxi,dydxi,dxdeta,dydeta]=to4DTensor(batch)
		optimizer.zero_grad()
		input=torch.cat([u_lf,v_lf],axis=1)
		input=input*(1+inputRand)
		if epoch==1:
			np.savetxt('InputU.txt',input[0,0,:,:].detach().cpu().numpy())
			np.savetxt('InputV.txt',input[0,1,:,:].detach().cpu().numpy())
		BICUBICU=model.US(input)[0,0,:,:].detach().cpu().numpy()
		BICUBICV=model.US(input)[0,1,:,:].detach().cpu().numpy()
		output=model(input)#model(cord)
		output_pad=udfpad(output)
		VTEMP=torch.zeros([1,1,ny_hf,nx_hf]).float().to('cuda')
		VSparseOBS=torch.zeros([1,1,ny_hf,nx_hf]).float().to('cuda')
		outputV_tmep=output_pad[:,1,:,:].reshape(output_pad.shape[0],1,
			                                output_pad.shape[2],
			                                output_pad.shape[3])
		for ii in range(len(idx1)):
			VTEMP[0,0,idx1[ii],idx2[ii]]=OFV_sb_cuda[0,0,idx1[ii],idx2[ii]]-outputV_tmep[0,0,idx1[ii],idx2[ii]]
			VSparseOBS[0,0,idx1[ii],idx2[ii]]=OFV_sb_cuda[0,0,idx1[ii],idx2[ii]]
		outputU=output_pad[:,0,:,:].reshape(output_pad.shape[0],1,
			                                output_pad.shape[2],
			                                output_pad.shape[3])
		
		outputV=VTEMP+output_pad[:,1,:,:].reshape(output_pad.shape[0],1,
			                                output_pad.shape[2],
			                                output_pad.shape[3])
		outputP=output_pad[:,2,:,:].reshape(output_pad.shape[0],1,
			                                output_pad.shape[2],
			                                output_pad.shape[3])
		XR=torch.zeros([batchSize,1,ny_hf,nx_hf]).to('cuda')
		YR=torch.zeros([batchSize,1,ny_hf,nx_hf]).to('cuda')
		MR=torch.zeros([batchSize,1,ny_hf,nx_hf]).to('cuda')
		TR=torch.zeros([batchSize,1,ny_hf,nx_hf]).to('cuda')
		for j in range(batchSize):
			#Impose BC
			outputU[j,0,-padSingleSide:,padSingleSide:-padSingleSide]=output[j,0,-1,:].reshape(1,nx_hf-2*padSingleSide) # up outlet bc zero gradient 
			outputU[j,0,:padSingleSide,padSingleSide:-padSingleSide]=0  # down inlet bc
			outputU[j,0,padSingleSide:-padSingleSide,-padSingleSide:]=0 # right wall bc			
			outputU[j,0,padSingleSide:-padSingleSide,0:padSingleSide]=0 # left  wall bc
			
			#outputU[j,0,0,0]=0.5*(outputU[j,0,0,1]+outputU[j,0,1,0])
			#outputU[j,0,0,-1]=0.5*(outputU[j,0,0,-2]+outputU[j,0,1,-1])
			outputU[j,0,0,0]=1*(outputU[j,0,0,1])
			outputU[j,0,0,-1]=1*(outputU[j,0,0,-2])
			

			outputV[j,0,-padSingleSide:,padSingleSide:-padSingleSide]=output[j,1,-1,:].reshape(1,nx_hf-2*padSingleSide) # up outlet bc zero gradient 
			outputV[j,0,:padSingleSide,padSingleSide:-padSingleSide]=torch.abs(model.source)			# down inlet bc
			einfer=torch.sqrt(criterion(infertruth,torch.abs(model.source))/criterion(torch.abs(model.source),torch.abs(model.source)*0))
			#print('Infer velocity====================',model.source[0])
			try:
				print('>>>>>>>Infer error<<<<<<< =======================',einfer.item())
				print('>>>>>>>model source<<<<<<< =======================',model.source)
			except:
				pass
			outputV[j,0,padSingleSide:-padSingleSide,-padSingleSide:]=0 					    # right wall bc			
			outputV[j,0,padSingleSide:-padSingleSide,0:padSingleSide]=0 					    # left  wall bc
			
			#outputV[j,0,0,0]=0.5*(outputV[j,0,0,1]+outputV[j,0,1,0])
			#outputV[j,0,0,-1]=0.5*(outputV[j,0,0,-2]+outputV[j,0,1,-1])
			outputV[j,0,0,0]=1*(outputV[j,0,0,1])
			outputV[j,0,0,-1]=1*(outputV[j,0,0,-2])
			

			
			outputP[j,0,-padSingleSide:,padSingleSide:-padSingleSide]=0 								  # up outlet zero pressure 
			outputP[j,0,:padSingleSide,padSingleSide:-padSingleSide]=output[j,2,0,:].reshape(1,nx_hf-2*padSingleSide)      # down inlet zero gradient bc
			
			outputP[j,0,padSingleSide:-padSingleSide,-padSingleSide:]=output[j,2,:,-1].reshape(ny_hf-2*padSingleSide,1)    # right wall zero gradient bc			
			outputP[j,0,padSingleSide:-padSingleSide,0:padSingleSide]=output[j,2,:,0].reshape(ny_hf-2*padSingleSide,1)     # left  wall zero gradient bc
			
			#outputP[j,0,0,0]=0.5*(outputP[j,0,0,1]+outputP[j,0,1,0])
			#outputP[j,0,0,-1]=0.5*(outputP[j,0,0,-2]+outputP[j,0,1,-1])
			outputP[j,0,0,0]=1*(outputP[j,0,0,1])
			outputP[j,0,0,-1]=1*(outputP[j,0,0,-2])
			
			
		
		dudx=dfdx(outputU,dydeta,dydxi,Jinv)
		d2udx2=dfdx(dudx,dydeta,dydxi,Jinv)
		
		
		dudy=dfdy(outputU,dxdxi,dxdeta,Jinv)
		d2udy2=dfdy(dudy,dxdxi,dxdeta,Jinv)
		
		


		dvdx=dfdx(outputV,dydeta,dydxi,Jinv)
		d2vdx2=dfdx(dvdx,dydeta,dydxi,Jinv)

		dvdy=dfdy(outputV,dxdxi,dxdeta,Jinv)
		d2vdy2=dfdy(dvdy,dxdxi,dxdeta,Jinv)

		dpdx=dfdx(outputP,dydeta,dydxi,Jinv)
		d2pdx2=dfdx(dpdx,dydeta,dydxi,Jinv)
		
		dpdy=dfdy(outputP,dxdxi,dxdeta,Jinv)
		d2pdy2=dfdy(dpdy,dxdxi,dxdeta,Jinv)


		#scalar transport 
		dTdx=dfdx(T_hardimpo,dydeta,dydxi,Jinv)
		d2Tdx2=dfdx(dTdx,dydeta,dydxi,Jinv)
		dTdy=dfdy(T_hardimpo,dxdxi,dxdeta,Jinv)
		d2Tdy2=dfdy(dTdy,dxdxi,dxdeta,Jinv)



		#Calculate PDE Residual
		continuity=(dudx+dvdy);
		#continuity=-(d2pdx2+d2pdy2)-d2udx2-d2vdy2-2*dudy*dvdx
		#u*dudx+v*dudy
		momentumX=outputU*dudx+outputV*dudy
		#-dpdx+nu*lap(u)
		forceX=-dpdx+nu*(d2udx2+d2udy2)
		# Xresidual
		Xresidual=(momentumX-forceX)


		#u*dvdx+v*dvdy
		momentumY=outputU*dvdx+outputV*dvdy
		#-dpdy+nu*lap(v)
		forceY=-dpdy+nu*(d2vdx2+d2vdy2)
		Yresidual=(momentumY-forceY)

		
		
		#T*dudx + u*dTdx
		convecX=T_hardimpo*dudx+outputU*dTdx
		#T*dvdy + v*dTdy
		convecY=T_hardimpo*dvdy+outputV*dTdy
		#nu*laplacian
		diffusionT=nu*d2Tdy2+nu*d2Tdx2
		Tresidual=(convecY+convecX-diffusionT)*0

		


		


		
		loss=(criterion(Xresidual,Xresidual*0)+\
		  criterion(Yresidual,Yresidual*0)+\
		  criterion(continuity,continuity*0)+\
		  criterion(Tresidual,Tresidual*0))
		loss.backward()
		optimizer.step()

		# Print return value
		loss_xm=criterion(Xresidual, Xresidual*0)
		loss_ym=criterion(Yresidual, Yresidual*0)
		loss_mass=criterion(continuity, continuity*0)
		loss_T=criterion(Tresidual, Tresidual*0)
		xRes+=loss_xm.item()
		yRes+=loss_ym.item()
		mRes+=loss_mass.item()
		TRes+=loss_T.item()
		CNNUNumpy=outputU[0,0,:,:].cpu().detach().numpy()
		CNNVNumpy=outputV[0,0,:,:].cpu().detach().numpy()
		CNNPNumpy=outputP[0,0,:,:].cpu().detach().numpy()
		eU=eU+np.sqrt(calMSE(OFU_sb,CNNUNumpy)/calMSE(OFU_sb,OFU_sb*0))
		eV=eV+np.sqrt(calMSE(OFV_sb,CNNVNumpy)/calMSE(OFV_sb,OFV_sb*0))
		eP=eP+np.sqrt(calMSE(OFP_sb,CNNPNumpy)/calMSE(OFP_sb,OFP_sb*0))
		eVmag=np.sqrt(calMSE(np.sqrt(OFU_sb**2+OFV_sb**2),np.sqrt(CNNUNumpy**2+CNNVNumpy**2))/calMSE(np.sqrt(OFU_sb**2+OFV_sb**2),np.sqrt(OFU_sb**2+OFV_sb**2)*0))
		eVBICUIC=np.sqrt(calMSE(np.sqrt(OFU_sb[1:-1,1:-1]**2+OFV_sb[1:-1,1:-1]**2),np.sqrt(BICUBICU**2+BICUBICV**2))/calMSE(np.sqrt(OFU_sb[1:-1,1:-1]**2+OFV_sb[1:-1,1:-1]**2),np.sqrt(OFU_sb[1:-1,1:-1]**2+OFV_sb[1:-1,1:-1]**2)*0))
		print('VelMagError_CNN=',eVmag)
		print('VelMagError_BI=',eVBICUIC)
		print('P_err_CNN=',np.sqrt(calMSE(OFP_sb,CNNPNumpy)/calMSE(OFP_sb,OFP_sb*0)))
	print('Epoch is ',epoch)
	print("xRes Loss is", (xRes/len(training_data_loader)))
	print("yRes Loss is", (yRes/len(training_data_loader)))
	print("mRes Loss is", (mRes/len(training_data_loader)))
	print("TRes Loss is", (TRes/len(training_data_loader)))
	print("eU Loss is", (eU/len(training_data_loader)))
	print("eV Loss is", (eV/len(training_data_loader)))
	print("eP Loss is", (eP/len(training_data_loader)))
	#TOL=[0.07,0.06,0.05,0.04,0.03,0.02,0.01]
	if epoch==1:
		np.savetxt('BIErrorVmag.txt',eVBICUIC*np.ones([4,4]))
	if (einfer.item()<0.05) or epoch%nEpochs==0 or epoch%5000==0 or epoch==100: # eP<0.15 and eVmag<0.04 and epoch==100 or epoch%5000==0 or epoch%nEpochs==0 or 
		torch.save(model, str(epoch)+'.pth')
		fig0=plt.figure()
		ax=plt.subplot(2,3,2)
		_,cbar=visualize2D(ax,mesh_hf.x,
			                  mesh_hf.y,
			           np.sqrt(outputU[0,0,:,:].cpu().detach().numpy()**2+\
			           		   outputV[0,0,:,:].cpu().detach().numpy()**2),'vertical',[0,1.0])
		setAxisLabel(ax,'p')
		ax.set_title('CNN')
		cbar.set_ticks([0,0.25,0.5,0.75,1.0])
		#ax.set_aspect('equal')

		ax=plt.subplot(2,3,3)
		_,cbar=visualize2D(ax,mesh_hf.x,
			           		  mesh_hf.y,
			           np.sqrt(OFU_sb**2+\
			           		   OFV_sb**2),'vertical',[0,1.0])
		cbar.set_ticks([0,0.25,0.5,0.75,1.0])
		setAxisLabel(ax,'p')
		ax.set_title('Truth')
		#ax.set_aspect('equal')

		ax=plt.subplot(2,3,4)
		_,cbar=visualize2D(ax,mesh_hf.x,
			           		  mesh_hf.y,
			                  2*VSparseOBS[0,0,:,:].cpu().detach().numpy(),'vertical',[0,1.0])
		cbar.set_ticks([0,0.25,0.5,0.75,1.0])
		setAxisLabel(ax,'p')
		ax.set_title('Observation')
		#ax.set_aspect('equal')

		ax=plt.subplot(2,3,1)
		_,cbar=visualize2D(ax,mesh_lf.x,
			           		  mesh_lf.y,
			           np.sqrt(input[0,0,:,:].cpu().detach().numpy()**2+\
			           		   input[0,1,:,:].cpu().detach().numpy()**2**2),'vertical',[0,1.0])
		cbar.set_ticks([0,0.25,0.5,0.75,1.0])
		setAxisLabel(ax,'p')
		ax.set_title('Input')
		#ax.set_aspect('equal')

		ax=plt.subplot(2,3,5)
		#pdb.set_trace()
		_,cbar=visualize2D(ax,mesh_hf.x[1:-1,1:-1],
			           		  mesh_hf.y[1:-1,1:-1],
			           		  np.sqrt(BICUBICU**2+BICUBICV**2),
							  'vertical',[0,1.0])
		cbar.set_ticks([0,0.25,0.5,0.75,1.0])
		setAxisLabel(ax,'p')
		ax.set_title('Bicubic')
		#ax.set_aspect('equal')
		

		'''
		ax=plt.subplot(2,3,4)
		visualize2D(ax,mesh_hf.x,
			           mesh_hf.y,
			           outputP[0,0,:,:].cpu().detach().numpy(),'vertical',[0,0.35])
		setAxisLabel(ax,'p')
		ax.set_title('Super-resolved '+'Pressure')

		ax=plt.subplot(2,3,5)
		visualize2D(ax,mesh_hf.x,
			           mesh_hf.y,
			           OFP_sb[:,:],'vertical',[0,0.35])
		setAxisLabel(ax,'p')
		ax.set_title('True '+'Pressure')
		'''

		ax_=plt.subplot(2,3,6)
		
		ax_.plot(mesh_hf.x[0,1:-1].reshape([1,47]),model.source.cpu().detach().numpy(),'x',label='Inferred',color='blue')
		ax_.plot(mesh_hf.x[0,:],velofunc(mesh_hf.x[0,:]),'--',label='True')
		setAxisLabel(ax_,'p')
		ax_.set_ylabel(r'$v$')
		ax_.set_title('Inlet '+'Velocity Profile')
		


		fig0.tight_layout(pad=1)
		fig0.savefig(str(epoch)+'Transport.pdf',bbox_inches='tight')
		plt.close(fig0)
		



	return (xRes/len(training_data_loader)), (yRes/len(training_data_loader)),\
		   (mRes/len(training_data_loader)), (TRes/len(training_data_loader)) ,(eU/len(training_data_loader)),\
		   (eV/len(training_data_loader)), (eP/len(training_data_loader)),model.source.detach().cpu().numpy(),einfer.item()
		
			
			







			
			

XRes=[];YRes=[];MRes=[];CRes=[]
EU=[];EV=[];EP=[]
Iinlet=[]
TotalstartTime=time.time()
EINFER=[]
for epoch in range(1,nEpochs+1):
	tic=time.time()
	xres,yres,mres,cres,eu,ev,ep,infer,einferr=train(epoch)
	print('Time of this epoch=',time.time()-tic)
	EINFER.append(einferr)
	XRes.append(xres)
	YRes.append(yres)
	MRes.append(mres)
	CRes.append(cres)
	EU.append(eu)
	EV.append(ev)
	EP.append(ep)
	Iinlet.append(infer)
	if einferr<0.01:
		break
TimeSpent=time.time()-TotalstartTime

plt.figure()
plt.plot(XRes,'-o',label='X-momentum Residual')
plt.plot(YRes,'-x',label='Y-momentum Residual')
plt.plot(MRes,'-*',label='Continuity Residual')
plt.plot(CRes,'-.',label='Transport Residual')
plt.xlabel('Epoch')
plt.ylabel('Residual')
plt.legend()
plt.yscale('log')
plt.savefig('convergence.pdf',bbox_inches='tight')
tikzplotlib.save('convergence.tikz')

plt.figure()
plt.plot(EU,'-o',label=r'$u$')
plt.plot(EV,'-x',label=r'$v$')
plt.plot(EP,'-*',label=r'$p$')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.yscale('log')
plt.savefig('error.pdf',bbox_inches='tight')
tikzplotlib.save('error.tikz')
EU=np.asarray(EU)
EV=np.asarray(EV)
EP=np.asarray(EP)
XRes=np.asarray(XRes)
YRes=np.asarray(YRes)
MRes=np.asarray(MRes)
CRes=np.asarray(CRes)
Iinlet=np.asarray(Iinlet)
np.savetxt('EU.txt',EU)
np.savetxt('EV.txt',EV)
np.savetxt('EP.txt',EP)
np.savetxt('Iinlet.txt',Iinlet.squeeze())
np.savetxt('XRes.txt',XRes)
np.savetxt('YRes.txt',YRes)
np.savetxt('MRes.txt',MRes)
np.savetxt('CRes.txt',CRes)
np.savetxt('TimeSpent.txt',np.zeros([2,2])+TimeSpent)
np.savetxt('EINFER.txt',np.asarray(EINFER))






















































'''
			dudx=Jinv[j:j+1,0:1,:,:]*(model.convdxi(outputU[j:j+1,0:1,:,:])*dydeta[j:j+1,0:1,:,:]-\
			     model.convdeta(outputU[j:j+1,0:1,:,:])*dydxi[j:j+1,0:1,:,:])
			d2udx2=Jinv[j:j+1,0:1,2:-2,2:-2]*(model.convdxi(dudx)*dydeta[j:j+1,0:1,2:-2,2:-2]-\
			       model.convdeta(dudx)*dydxi[j:j+1,0:1,2:-2,2:-2])
			dvdx=Jinv[j:j+1,0:1,:,:]*(model.convdxi(outputV[j:j+1,0:1,:,:])*dydeta[j:j+1,0:1,:,:]-\
			     model.convdeta(outputV[j:j+1,0:1,:,:])*dydxi[j:j+1,0:1,:,:])
			d2vdx2=Jinv[j:j+1,0:1,2:-2,2:-2]*(model.convdxi(dvdx)*dydeta[j:j+1,0:1,2:-2,2:-2]-\
			       model.convdeta(dvdx)*dydxi[j:j+1,0:1,2:-2,2:-2])

			dudy=Jinv[j:j+1,0:1,:,:]*(model.convdeta(outputU[j:j+1,0:1,:,:])*dxdxi[j:j+1,0:1,:,:]-\
			     model.convdxi(outputU[j:j+1,0:1,:,:])*dxdeta[j:j+1,0:1,:,:])
			d2udy2=Jinv[j:j+1,0:1,2:-2,2:-2]*(model.convdeta(dudy)*dxdxi[j:j+1,0:1,2:-2,2:-2]-\
			     model.convdxi(dudy)*dxdeta[j:j+1,0:1,2:-2,2:-2])
			dvdy=Jinv[j:j+1,0:1,:,:]*(model.convdeta(outputV[j:j+1,0:1,:,:])*dxdxi[j:j+1,0:1,:,:]-\
			     model.convdxi(outputV[j:j+1,0:1,:,:])*dxdeta[j:j+1,0:1,:,:])
			d2vdy2=Jinv[j:j+1,0:1,2:-2,2:-2]*(model.convdeta(dvdy)*dxdxi[j:j+1,0:1,2:-2,2:-2]-\
			     model.convdxi(dvdy)*dxdeta[j:j+1,0:1,2:-2,2:-2])

			dpdx=Jinv[j:j+1,0:1,:,:]*(model.convdxi(outputP[j:j+1,0:1,:,:])*dydeta[j:j+1,0:1,:,:]-\
			     model.convdeta(outputP[j:j+1,0:1,:,:])*dydxi[j:j+1,0:1,:,:])
			dpdy=Jinv[j:j+1,0:1,:,:]*(model.convdeta(outputP[j:j+1,0:1,:,:])*dxdxi[j:j+1,0:1,:,:]-\
			     model.convdxi(outputP[j:j+1,0:1,:,:])*dxdeta[j:j+1,0:1,:,:])

			continuity=dudx[:,:,2:-2,2:-2]+dudy[:,:,2:-2,2:-2];
			#u*dudx+v*dudy
			momentumX=outputU[j:j+1,:,2:-2,2:-2]*dudx+\
			          outputV[j:j+1,:,2:-2,2:-2]*dvdx
			#-dpdx+nu*lap(u)
			forceX=-dpdx[0:,0:,2:-2,2:-2]+nu*(d2udx2+d2udy2)
			# Xresidual
			Xresidual=momentumX[0:,0:,2:-2,2:-2]-forceX   

			#u*dvdx+v*dvdy
			momentumY=outputU[j:j+1,:,2:-2,2:-2]*dvdx+\
			          outputV[j:j+1,:,2:-2,2:-2]*dvdy
			#-dpdy+nu*lap(v)
			forceY=-dpdy[0:,0:,2:-2,2:-2]+nu*(d2vdx2+d2vdy2)
			# Yresidual
			Yresidual=momentumY[0:,0:,2:-2,2:-2]-forceY 
			'''