from pathlib import Path
import ast
import numpy as np
import pandas as pd
from scipy.linalg import lu_factor,lu_solve
import argparse
parser = argparse.ArgumentParser(description='Independent GVA and FCE reconstruction from EXIOBASE raw TSV files.')
parser.add_argument('exiobase', type=Path, help='Directory containing EXIOBASE 3.9.6 IOT_2022_ixi')
parser.add_argument('allocation', type=Path, help='Path to the reviewed branch’s 2b_PB_Allocation directory')
args = parser.parse_args()
D = args.exiobase
P = args.allocation
xdata=pd.read_csv(D/'x.txt',sep='\t')
x=xdata.iloc[:,2].to_numpy(); regions=xdata.region.to_numpy(); labels=list(zip(regions,xdata.sector)); n=len(x)
j=labels.index(('GB','Chemicals nec'))
vdat=pd.read_csv(D/'factor_inputs/F.txt',sep='\t',header=[0,1],index_col=0)
assert list(vdat.columns)==labels
sel=[k for k in vdat.index if k=='Other net taxes on production' or k.startswith('Compensation of employees') or k in ('Operating surplus: Consumption of fixed capital','Operating surplus: Remaining net operating surplus')]
V=vdat.loc[sel].values.sum(0)
Vall=vdat.sum(0).to_numpy()
av=lambda z: np.divide(z,x,out=np.zeros_like(x),where=x!=0)
source=ast.parse((P/'vendor/pbaesa/pbaesa/allocation.py').read_text())
pop=ast.literal_eval(next(i for i in ast.walk(source) if isinstance(i,ast.Assign) and isinstance(i.targets[0],ast.Name) and i.targets[0].id=='sPOPr_dict' and isinstance(i.value,ast.Dict)).value)
Y=pd.read_csv(D/'Y.txt',sep='\t',header=[0,1],index_col=[0,1])
assert list(Y.index)==labels
FF=Y.loc[:,Y.columns.get_level_values(1).str.startswith('Final consumption expenditure')].T.groupby(level=0).sum().T
FR=FF/FF.sum(); direct=FR.to_numpy()@np.array([pop[r] for r in FR.columns])
e=np.eye(1,n,j).ravel()
with open(D/'A.txt','rb') as f:
    rh=f.readline().decode().rstrip().split('\t')[2:]
    sh=f.readline().decode().rstrip().split('\t')[2:]
    assert labels==list(zip(rh,sh))
    next(f)
    A=np.empty((n,n),order='F')
    for k,line in enumerate(f):
        r,s,row=line.split(b'\t',2)
        assert (r.decode(),s.decode())==labels[k]
        A[k]=np.fromstring(row,sep='\t')
Z=A[:,j].copy()*x[j]
print('raw input',n,len(sel),sum(pop.values()),'x,V,j',x[j],V[j],j,flush=True)
A*=-1
A.flat[::n+1]+=1
lu,perm=lu_factor(A,overwrite_a=True,check_finite=False)
col=lu_solve((lu,perm),e,check_finite=False)
row,c=np.transpose(lu_solve((lu,perm),np.c_[e,np.ones(n)],trans=1,check_finite=False))
R=sorted(pop)
rg=lambda a:np.array([sum(a[regions==r]) for r in R])
G=rg(V); p=np.array([pop[r] for r in R])
Gtotal=x[j]*av(V)@col
T=Z.sum()+V[j]; alpha=(rg(Z)+np.array([V[j] if r=='GB' else 0 for r in R]))/T
factor=lambda T,alpha:(alpha/G)@p*T
bug=V[j]*x[j]*(av(np.ones(n))@col)
full=rg(av(V)*col)*x[j]
print('Gtotal, intensity, m, T, pGB, GGB, alphaGB, AF, bugT, bugAF, fullAF, fullGB',Gtotal,Gtotal/x[j],Gtotal/V[j],T,p[R.index('GB')],G[R.index('GB')],alpha[R.index('GB')],factor(Gtotal,alpha),bug,factor(bug,alpha),sum(full/G*p),full[R.index('GB')]/Gtotal,flush=True)
print('all_inputs vs six intensity, alternative accounting',av(Vall)@col, av(Vall-V)@col)
S=row*c[j]/col[j]/c
print('FCE direct',direct[j],'own S',S[j],'c[j],Ljj',c[j],col[j],'formula FCE',S@direct,'code replay',(S.sum()+1)*direct[j],'double diagonal with corrected axis',S@direct+direct[j],'single diagonal vendor axis',S.sum()*direct[j],flush=True)
print('sector FCE published workbook matching row')
from openpyxl import load_workbook
w=load_workbook(P/'data/Allocation Factors_2022.xlsx',read_only=True,data_only=True)
for sheet in w:
 for i,r in enumerate(sheet.values):
  if i<2 or ('GB' in r and 'Chemicals nec' in r) or any('GB_Chemicals nec'==s for s in r): print(sheet.title,r[:12])
import json
net=861*4.65*1000*92.09802/1e6/1.2766*(90.6/81.1)/.85276/x[j]
impact=pd.read_csv(P/'results_final/annual_impacts.csv').query("case_id=='two_stage' and category=='Climate Change'").annual_impact.item()
d=json.loads((P/'data/ability_to_pay_weights.json').read_text())['cases']['primary_2022_MER']['atp_weight_by_region']
p_atp=np.array([d[r] for r in FR.columns]); fce_atp=S@(FR.to_numpy()@p_atp)
for name,af in [('FCE',S@direct),('FCE sector weights normalised',S@direct/sum(pop.values())),('GVA',factor(Gtotal,alpha)),('full GVA',p@(full/G)),('ATP FCE using branch weight',fce_atp),('ATP full GVA using branch weight',p_atp@(full/G))]:
 print(name,af,'net fraction',net,'budget',af*net,'utilisation',impact/(af*net)*100)
print('GB target allocation raw satellite detail',dict(vdat.iloc[:,j]))
print('FCE raw regional column mass error, local negative cells, min FCE, total directly negative, minimum c',max(abs(FR.sum().values-1)),(FF<0).sum().sum(),np.min(FF.to_numpy()),sum(direct<0),min(c))
print('FCE positive output / zero FCE',sum((direct==0)&(x>0)))
pure=np.where((direct==0)&(x>0))[0]
for i in pure:
 if regions[i]=='GB':
  ej=np.eye(1,n,i).ravel()
  lrow=lu_solve((lu,perm),ej,trans=1,check_finite=False)
  print('actual B2B counterexample',labels[i],direct[i],(lrow*c[i]/lrow[i]/c)@direct); break
i=pure[np.argmax(x[pure])]
lrow=lu_solve((lu,perm),np.eye(1,n,i).ravel(),trans=1,check_finite=False)
print('largest zero-direct row',labels[i],'output',x[i],'best-supported FCE',(lrow*c[i]/lrow[i]/c)@direct)
print('normalised GVA AF',factor(Gtotal,alpha)/sum(pop.values()))
