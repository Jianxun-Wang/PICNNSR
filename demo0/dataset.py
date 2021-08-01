from torch.utils.data import Dataset, DataLoader
import pdb
import numpy as np
class VaryGeoDataset(Dataset):
	"""docstring for hcubeMeshDataset"""
	def __init__(self,MeshList):
		self.MeshList=MeshList
	def __len__(self):
		return len(self.MeshList)
	def __getitem__(self,idx):
		mesh=self.MeshList[idx]
		x=mesh.x
		y=mesh.y
		xi=mesh.xi
		eta=mesh.eta
		J=mesh.J_ho
		Jinv=mesh.Jinv_ho
		dxdxi=mesh.dxdxi_ho
		dydxi=mesh.dydxi_ho
		dxdeta=mesh.dxdeta_ho
		dydeta=mesh.dydeta_ho
		cord=np.zeros([2,x.shape[0],x.shape[1]])
		cord[0,:,:]=x; cord[1,:,:]=y
		InvariantInput=np.zeros([2,J.shape[0],J.shape[1]])
		InvariantInput[0,:,:]=J
		InvariantInput[1,:,:]=Jinv
		return [InvariantInput,cord,xi,eta,J,
		        Jinv,dxdxi,dydxi,
		        dxdeta,dydeta]