#!usr/bin/env/python
#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets, cluster

digitos=datasets.load_digits()
datos_raw=digitos.data;
medias_datos_raw=np.average(datos_raw, axis=0);
datos_tot=datos_raw-medias_datos_raw;
z_Target=digitos.target;

def hallar_error(z1,z2):
	suma=0;
	for j in range(len(z1)):
		if z1[j]!=z2[j]:
			suma=suma+1;
		#fin if
	#fin for
	resp=suma/len(z1);
	return resp
#fin hallar_error

def hallar_equivalencia(z1,z2,nClusters):
	suma=0;
	for j in range(nClusters):
		vj=z2[z1==j];
		m_vj=np.bincount(vj).argsort()[-1];
		for l in range(len(vj)):
			if vj[l]!=m_vj:
				suma=suma+1;
			#fin if
		#fin fir
	#fin for
	resp=suma/len(z1);
	return resp
#fin hallar_equivalencia

def hallarPhi_PCA(Datos_m,nComp):
	nDatos=len(Datos_m[:,0]);
	nVar=len(Datos_m[0,:]);
	covarianzas=(1/(nDatos-1))*(Datos_m.T).dot(Datos_m);
	valPropios, vectPropios = np.linalg.eig(covarianzas);
	posiciones=np.argsort(valPropios)[::-1];
	phi=np.zeros((nVar,nComp));
	for k in range(nComp):
		phi[:,k]=vectPropios[:,posiciones[k]];
	#fin for 
	return phi;
#fin función

def reducir_PCA(Datos_m,nComp):
	phi=hallarPhi_PCA(Datos_m,nComp);
	xnuevo=Datos_m.dot(phi);
	return xnuevo;
#fin reducir PCA. 

datos_red=reducir_PCA(datos_tot,2);

def hallarCoordenadasCentroides(Datos,z,nCentroides):
	rCentroides=np.zeros((len(Datos[0,:]),nCentroides));
	for k in range(nCentroides):
		rCentroides[:,k]=np.average(Datos[z==k,:],axis=0);
	#fin for
	return rCentroides
#fin cunción

def  inicializarKpp(Datos,nCentroides):
	nDatos=len(Datos[:,0]);
	centroide1=np.random.choice(nDatos);
	Dm=Datos-Datos[centroide1,:];
	Dm2=Dm*Dm;
	r_2=np.sum(Dm2,axis=1);
	p_xj=r_2/sum(r_2);
	centroides=Datos[centroide1,:];
	for k in range(nCentroides-1):
		coord_centroideNuevo=np.random.choice(nDatos,p=p_xj);
		centroideNuevo=Datos[coord_centroideNuevo,:];
		centroides=np.c_[centroides,centroideNuevo];
	#fin for 
	return centroides;
#fin inicializar 

def sonIguales(array1, array2):
	#esta función mira si dos array son exactamente iguales
	resp=True;
	if len(array1)!=len(array2):
		print(".|. Los array tienen que ser iguales .|. ");
		print("Le botaré un False... por pendejo... ");
		resp=False;
	#fin if
	for i in range(len(array1)):
		if array1[i]!=array2[i]:
			resp=False;
		#fin if
	#fin for 
	return resp;
#fin función igualitos

def KMeansC(Datos,nCentroides,rkini=0,opcion=0):
	#llamaremos rk a las coordenadas de los centroides
	nDatos=len(Datos[:,0]);
	if opcion==0:
		rk=inicializarKpp(Datos,nCentroides);
	else:
		rk=rkini;
	#fin if
	MDatos=np.tile(Datos,(nCentroides,1,1));
	MDatosT=MDatos.transpose(1,2,0);
	cIter=0;
	maxIter=100;
	zDistintos=True;
	z0=np.zeros(nDatos);
	while zDistintos==True and cIter<=maxIter:
		M=MDatosT-rk;
		distancias=np.sum(M*M, axis=1);
		Args_d=np.argsort(distancias);
		z_nuevo=Args_d[:,0];
		rk=hallarCoordenadasCentroides(Datos,z_nuevo,nCentroides);
		if sonIguales(z0,z_nuevo):
			zDistintos=False;
		#fin if
		z0=z_nuevo;
		cIter=cIter+1;
	#fin whiletouch 
	if cIter>=maxIter:
		print("OJO!!! máximas iteraciones alcanzadas");
	#fin if 
	return [rk, z0];
#fin función

rkini=hallarCoordenadasCentroides(datos_tot,z_Target,10);
Resultados1=KMeansC(datos_tot,10,rkini,opcion=1);
Resultados2=KMeansC(datos_tot,10);
rk1=Resultados1[0];
z1=Resultados1[1];
rk2=Resultados2[0];
z2=Resultados2[1];

modelo3=cluster.KMeans(n_clusters=10, init=rkini.T, n_init=1);
Resultados3=modelo3.fit(datos_tot);
rk3=modelo3.cluster_centers_;
z3=modelo3.labels_;

modelo4=cluster.KMeans(n_clusters=10);
Resultados4=modelo4.fit(datos_tot);
rk4=modelo4.cluster_centers_;
z4=modelo4.labels_;

print("La diferencia entre tu y sklearn para centroides inicializados es: ", hallar_error(z1,z3));
print(np.c_[z1,z3]);
print("La diferencia entre tu y sklearn para centroides k++ es: ",hallar_equivalencia(z2,z4,10));
print(np.c_[z2,z4]);
