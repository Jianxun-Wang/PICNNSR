import Ofpp
import numpy as np
from pyMesh import hcubeMesh
def read_vessel_mesh(ofmeshfile):
	h=0.01
	OFBCCoord=Ofpp.parse_boundary_field(ofmeshfile)
	OFLOWC=OFBCCoord[b'low'][b'value']
	OFUPC=OFBCCoord[b'up'][b'value']
	OFLEFTC=OFBCCoord[b'left'][b'value']
	OFRIGHTC=OFBCCoord[b'rifht'][b'value']
	leftX=OFLEFTC[:,0];leftY=OFLEFTC[:,1]
	lowX=OFLOWC[:,0];lowY=OFLOWC[:,1]
	rightX=OFRIGHTC[:,0];rightY=OFRIGHTC[:,1]
	upX=OFUPC[:,0];upY=OFUPC[:,1]
	ny=len(leftX);nx=len(lowX)
	myMesh=hcubeMesh(leftX,leftY,rightX,rightY,
				 lowX,lowY,upX,upY,h,True,True,
				 tolMesh=1e-10,tolJoint=1e-1)
	return myMesh,nx,ny

def proj_solu_from_ofmesh_2_mymesh(OFPic,myMesh):
	OFX=OFPic[:,:,0]
	OFY=OFPic[:,:,1]
	OFU=OFPic[:,:,2]
	OFV=OFPic[:,:,3]
	OFP=OFPic[:,:,4]
	try:
		OFT=OFPic[:,:,5]
	except:
		pass
	OFU_sb=np.zeros(OFU.shape)
	OFV_sb=np.zeros(OFV.shape)
	OFP_sb=np.zeros(OFP.shape)
	try:
		OFT_sb=np.zeros(OFT.shape)
	except:
		pass
	ny,nx=myMesh.x.shape
	for i in range(nx):
		for j in range(ny):
			dist=(myMesh.x[j,i]-OFX)**2+(myMesh.y[j,i]-OFY)**2
			idx_min=np.where(dist == dist.min())
			OFU_sb[j,i]=OFU[idx_min]
			OFV_sb[j,i]=OFV[idx_min]
			OFP_sb[j,i]=OFP[idx_min]
			try:
				OFT_sb[j,i]=OFT[idx_min]
			except:
				pass

	try:
		return [OFU_sb, OFV_sb, OFP_sb,OFT_sb]
	except:
		return [OFU_sb, OFV_sb, OFP_sb]
