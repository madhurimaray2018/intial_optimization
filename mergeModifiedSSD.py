# coding: utf-8 -*-
#
# kernighan_lin.py - Kernighan–Lin bipartition algorithm
#
# Copyright 2011 Ben Edwards <bedwards@cs.unm.edu>.
# Copyright 2011 Aric Hagberg <hagberg@lanl.gov>.
# Copyright 2015 NetworkX developers.
#
# This file is part of NetworkX.
#
# NetworkX is distributed under a BSD license; see LICENSE.txt for more
# information.
"""Functions for computing the Kernighan–Lin bipartition algorithm."""
from __future__ import division

from collections import defaultdict
from itertools import islice
from operator import itemgetter
import random
import sys
import numpy
import math

import networkx as nx
from networkx.utils import not_implemented_for
from networkx.algorithms.community.community_utils import is_partition
import os
from itertools import izip
from operator import itemgetter, attrgetter

__all__ = ['kernighan_lin_bisection']
#---------------------------------------------------------Parameters--------------------------------------------------------------------
#Please set the parameters here
k=4  #Parameter   keep app no and chunk no relative prime
app_count=47 #4  #Prarameter  
total_ssd_in_rack=9 # this is number of ssd inside a rack
chunk_count=1500 #10 #Parameter
loop_count=3 # Parameter  loopcount must be less than total number of app then we will go to infinite loop
node_limit = 30*1000*1000 # approx 35% per node utilized  #Parameter (same here this could be bandwidth or io as above, not sure, assuming io for now)
large_node_limit=(total_ssd_in_rack-1)*node_limit # this is the rack level limit
pod_limit =((k*k*large_node_limit)/4)  #Parameter  (this could be io or bandwidth, not sure about the limit)
#node_cpu_limit = 99999999999 #Parameter
node_io_limit=node_limit #Parameter (Not Sure about definition)
baseAppRate=2560*1000*1000#Parameter (definition, this is the total io utilization of the network in terms of KBPS)
alpha=0.8 #alpha for zipf
seed=3 #Parameter  this is the seed for random number generation , incase you donot want to specify anything set it 0 then it will take timestamp
#---------------------------------------------------------------------------------------------------------------------------------------
pod_number=0
node_number=0
node_id=1
LL=[]
LP=[]
budget=node_limit
Data=[]
App=[]
total_node_fat_tree=(((k*k)*k)/4) # this is no parameter it is the total numbers of nodes in a fat tree
copyList=[]
def printApp2(L,nodeNumber):
     f2=open('./app.txt', 'a+')
     print>>f2, nodeNumber,
     for l in L:
        for p in l:
                print>>f2,",",
		print>>f2,int(p),
     print >>f2 ,"\n"
     f2.close()
     return

def printData2(L,nodeNumber):	
     f1=open('./data.txt', 'a+')
     print>>f1, nodeNumber,
     for l in L:
        for p in l:
                print>>f1,",",
		print>>f1,int(p),
     print >>f1 ,"\n"
     f1.close()
     return

def binPacking():
	global LP
	global k
	global budget
	LListA=[]
	LListD=[]
	maxval=budget
	p=0
	old_pod=-1
	LLP=[]
	last_pod=-1
	actualNode=0
	w, h = 4, 4;
	Matrix = [[0 for x in range(w)] for y in range(h)] 
	#print Matrix
	for d in LL:
		for weight,pod,node in zip(*[iter(d)]*3): 
			if old_pod==-1:
				old_pod=pod
			if(pod==old_pod):
				w=w+weight
				old_pod=pod
			else:
				P=[]
				P.append(old_pod) # then insert only one edge with that budget
			 	P.append(w)  
				LLP.append(P)
				w=0
				w=w+weight
				old_pod=pod
	P=[]
	P.append(pod) # then insert only one edge with that budget
 	P.append(w)  
	LLP.append(P)
	#print "LLP" , LLP
	LLP_sorted=sorted(LLP, key=itemgetter(1), reverse=True)	
	#print "LLP_sorted",LLP_sorted			
	for t in LLP_sorted:
		for l in LP:
			for weight,pod,node in zip(*[iter(l)]*3):  #pod and node is bascially for data tracking in app.txt and data.txt
				if(pod==t[0]):
					if last_pod==-1:
						last_pod=t[0]

					if(t[0]!=last_pod):
						sum=0
						for i in range(k):
							sum=sum + Matrix[p][i]	
						if sum<t[1] and p + 1 <k:
	                                               #write to the file before changing the pod
        	                                        printNode = ( p * k ) + actualNode
                	                                if LListD!=[]:
							 	printData2(LListD,printNode)
							if LListA!=[]:
                        	                        	printApp2(LListA,printNode)
                                	                LListA=[]
                                        	        LListD=[]
							#print("we are increasing the values here")
							p = p + 1
							maxval=budget
							actualNode=0	
					if maxval>=weight:
						#print "we are in here 1- alloting to the same node",  p,actualNode,maxval,weight
						maxval=maxval-weight
						Matrix[p][actualNode]=maxval  # we are registering the remaining in any node
						if(len(App[node])!=0):
							LListA.append(App[node])
						if(len(Data[node])!=0):
							LListD.append(Data[node])
						#print Matrix
						#then i can include the this under the same node
						#use this pod number or whatever is current and current node number
						last_pod=t[0]
					elif((actualNode+1)<k): #if the node limit exceeds - so we might be in the same pod but the limit of a node exceeds
						#before chaging the node write to the file
						#calculate the real node number
                                                printNode = ( p * k ) + actualNode
                                                if LListD!=[]:
	 	                                        printData2(LListD,printNode)
                                                if LListA!=[]:
                                                        printApp2(LListA,printNode)
	
					
						LListA=[]	
						LListD=[]
						actualNode=actualNode+1	
						maxval=budget	
						maxval=maxval-weight
						#print "we are in here 2- Node limit exceeded so creating another node in the same pod",p,actualNode,maxval,weight	
						Matrix[p][actualNode]=maxval
						if(len(App[node])!=0):
							LListA.append(App[node])
						if(len(Data[node])!=0):
							LListD.append(Data[node])	
						last_pod=t[0]
					elif (p+1<k): #means this is a new pod
						#write to the file before changing the pod
						printNode = ( p * k ) + actualNode
                                                if LListD!=[]:
                                                        printData2(LListD,printNode)
                                                if LListA!=[]:
                                                        printApp2(LListA,printNode)



                  				LListA=[]	
						LListD=[]
						actualNode=0 #lets just consider the actual node number of a pod starts for 0 later we can change this				
						maxval=budget
						p=p+1
						#print "we are in here 3 - going to the next pod",p,actualNode,maxval,weight
						maxval=maxval-weight
						if(len(App[node])!=0):
							LListA.append(App[node])
						if(len(Data[node])!=0):
							LListD.append(Data[node])
						#print Matrix
						#print p				
						Matrix[p][actualNode]=maxval
						last_pod=t[0]
	
					else: # we run out of the total number of nodes in a pod
						#look thoroughout the entire pod first
						#print "we are in here 4 worst case scenario we have to look through the entire sets of node to place the item", p
						for i in range(k):  #this is special case , file writing is not done for this - need to inspect it
								for j in range(k):
									if(Matrix[i][j]>=weight):
										Matrix[i][j]=Matrix[i][j]-weight   
									else:
										"not placed weight",weight,p
							
									#look into all the allocated nodes in the fat tree
									#if not then go to the next pod of the fat tree which is a little risky

				#else:
					#print "no"
			#LP.pop(s)
	
	#print Matrix			
	if LListD!=[]:
        	printNode = ( p * k ) + actualNode
        	printData2(LListD,printNode)
	if LListA!=[]:
		printNode = ( p * k ) + actualNode
        	printApp2(LListA,printNode)
	
	return

def clean_all():
#removing existing files
	cwd=os.getcwd()

	myfile="/final.txt"
	path=cwd+myfile	
	if os.path.isfile(path):
		os.remove(path)

	myfile="/node.txt"
        path=cwd+myfile
        if os.path.isfile(path):
                os.remove(path)

	myfile="/edge.txt"
        path=cwd+myfile
        if os.path.isfile(path):
                os.remove(path)

	myfile="/app.txt"
        path=cwd+myfile
        if os.path.isfile(path):
                os.remove(path)

	myfile="/data.txt"
        path=cwd+myfile
        if os.path.isfile(path):
                os.remove(path)
	
 	myfile="/appfreq.txt"
        path=cwd+myfile
        if os.path.isfile(path):
                os.remove(path)



def _compute_delta(G, A, B, weight):
    # helper to compute initial swap deltas for a pass
    delta = defaultdict(float)
    for u, v, d in G.edges(data=True):
        w = d.get(weight, 1)
        if u in A:
            if v in A:
                delta[u] -= w
                delta[v] -= w
            elif v in B:
                delta[u] += w
                delta[v] += w
        elif u in B:
            if v in A:
                delta[u] += w
                delta[v] += w
            elif v in B:
                delta[u] -= w
                delta[v] -= w
    return delta


def _update_delta(delta, G, A, B, u, v, weight):
    # helper to update swap deltas during single pass
    for _, nbr, d in G.edges(u, data=True):
        w = d.get(weight, 1)
        if nbr in A:
            delta[nbr] += 2 * w
        if nbr in B:
            delta[nbr] -= 2 * w
    for _, nbr, d in G.edges(v, data=True):
        w = d.get(weight, 1)
        if nbr in A:
            delta[nbr] -= 2 * w
        if nbr in B:
            delta[nbr] += 2 * w
    return delta


def _kernighan_lin_pass(G, A, B, weight):
    # do a single iteration of Kernighan–Lin algorithm
    # returns list of  (g_i,u_i,v_i) for i node pairs u_i,v_i
    multigraph = G.is_multigraph()
    delta = _compute_delta(G, A, B, weight)
    swapped = set()
    gains = []
    while len(swapped) < len(G):
        gain = []
        for u in A - swapped:
            for v in B - swapped:
                try:
                    if multigraph:
                        w = sum(d.get(weight, 1) for d in G[u][v].values())
                    else:
                        w = G[u][v].get(weight, 1)
                except KeyError:
                    w = 0
                gain.append((delta[u] + delta[v] - 2 * w, u, v))
        if len(gain) == 0:
            break
        maxg, u, v = max(gain, key=itemgetter(0))
        swapped |= {u, v}
        gains.append((maxg, u, v))
        delta = _update_delta(delta, G, A - swapped, B - swapped, u, v, weight)
    return gains



def node_sum(G,A):
     suma=0
     for (u,wt) in G.nodes.data('weight'):
        if u in set(A):
                suma=suma+wt
     return suma



def node_sum2(G): # ths function currently is not in use
     suma=0
     for (u,wt) in G.nodes.data('weight'):
                suma=suma+wt
     return suma


@not_implemented_for('directed')
def kernighan_lin_bisection(G, partition=None, max_iter=10, weight='weight'):
    """Partition a graph into two blocks using the Kernighan–Lin
    algorithm.

    This algorithm paritions a network into two sets by iteratively
    swapping pairs of nodes to reduce the edge cut between the two sets.

    Parameters
    ----------
    G : graph

    partition : tuple
        Pair of iterables containing an initial partition. If not
        specified, a random balanced partition is used.

    max_iter : int
        Maximum number of times to attempt swaps to find an
        improvemement before giving up.

    weight : key
        Edge data key to use as weight. If None, the weights are all
        set to one.

    Returns
    -------
    partition : tuple
        A pair of sets of nodes representing the bipartition.

    Raises
    -------
    NetworkXError
        If partition is not a valid partition of the nodes of the graph.

    References
    ----------
    .. [1] Kernighan, B. W.; Lin, Shen (1970).
       "An efficient heuristic procedure for partitioning graphs."
       *Bell Systems Technical Journal* 49: 291--307.
       Oxford University Press 2011.

    """
    # If no partition is provided, split the nodes randomly into a
    # balanced partition.
    if partition is None:
        nodes = list(G)
        random.shuffle(nodes)
        h = len(nodes) // 2
        partition = (nodes[:h], nodes[h:])
    # Make a copy of the partition as a pair of sets.
    try:
        A, B = set(partition[0]), set(partition[1])
    except:
        raise ValueError('partition must be two sets')
    if not is_partition(G, (A, B)):
        raise nx.NetworkXError('partition invalid')
    for i in range(max_iter):
        # `gains` is a list of triples of the form (g, u, v) for each
        # node pair (u, v), where `g` is the gain of that node pair.
        gains = _kernighan_lin_pass(G, A, B, weight)
        csum = list(nx.utils.accumulate(g for g, u, v in gains))
	#if csum==[]:
	#	print (A)
	#	print (B)
	#	return A,B
        max_cgain = max(csum)
        if max_cgain <= 0:
	    break
        # Get the node pairs up to the index of the maximum cumulative
        # gain, and collect each `u` into `anodes` and each `v` into
        # `bnodes`, for each pair `(u, v)`.
        index = csum.index(max_cgain)
        nodesets = islice(zip(*gains[:index + 1]), 1, 3)
        anodes, bnodes = (set(s) for s in nodesets)
        A |= bnodes
        A -= anodes
        B |= anodes
        B -= bnodes
    return A, B



def create_graph_from_set(G,A):
     Ga = nx.MultiGraph()
     app = []
     data = []
####creating two graphs from the above A and B node sets
     for (u, v, wt) in G.edges.data('weight'):
	if u in set(A) and v in set(A) :
		Ga.add_weighted_edges_from([(u, v, wt)])
     for (u,wt) in G.nodes.data('weight'):
	if u in set(A): 
		Ga.add_node(u,weight=wt,app=app,data=data)
     for (u, app) in G.nodes.data('app'):
        if u in set(A):
		Ga.nodes[u]['app']=app
		#print(Ga.nodes[u])
     for (u, data) in G.nodes.data('data'):
        if u in set(A):
		Ga.nodes[u]['data']=data
     #print(Ga.nodes.data())        
     return Ga




def printApp3(G,nodeNumber):
     global pod_number
     f2=open('./app.txt', 'a+')
     print>>f2, nodeNumber,# ",",  pod_number,
     for (u, app) in G.nodes.data('app'):
	#print>>f2,u,"-",
        for p in G.nodes[u]['app']:
                print>>f2,",",
		print>>f2,int(p),
     print >>f2 ,"\n"
     f2.close()
     return



def printData3(G,nodeNumber):	
     global pod_number,copyList
     global total_ssd_in_rack
     ssd_number=(total_ssd_in_rack*node_number)+1
     f1=open('./data.txt', 'a+')
     #print>>f1, nodeNumber,","# pod_number ,
     for (u, data) in G.nodes.data('data'):
       print>>f1,ssd_number,
       ssd_number=ssd_number+1
       for p in G.nodes[u]['data']:
                print>>f1,",",
		flag=0
		for cL in copyList:
			if int(cL[1])==int(p):
				print>>f1,int(cL[0]),
				flag=1
				break
		if flag==0:
			print>>f1,int(p),
       print >>f1 ,"\n"
     print >>f1 ,"\n"
     f1.close()
     return

def process_node(G,A):
     global node_number,large_node_limit
     global k
     global pod_number
     global node_limit #= sys.argv[2]
     global LL
     #d=k/2
     #limit=d*d*k
     #print(A)
     Ga = create_graph_from_set(G,A)
     suma=node_sum(G,A)
     if(suma>int(large_node_limit)):
	#print "node limit exceeded",suma
        P,Q=kernighan_lin_bisection(Ga)
        process_node(Ga,P)
        process_node(Ga,Q)
     else:
	#print "node limit not exceeded",suma,node_number,pod_number
        printData3(Ga,node_number)
        printApp3(Ga,node_number) 
        #P=[]
	#P.append(suma) # then insert only one edge with that budget
 	#P.append(pod_number) 
	#P.append(node_number) 
	#LL.append(P)
	node_number=node_number+1
        #assert((k * (pod_number)<=node_number) and (k*(pod_number+1))>node_number )
        return


def process(G,A):
     global pod_number, node_number, large_node_limit
     global pod_limit 
     global node_limit 
     global k 
     node_number=pod_number*int(k)
     #node_number_data=node_number
     #node_number_app=node_number
     Ga = create_graph_from_set(G,A)
     suma=node_sum(G,A)
     if(suma>int(pod_limit)):
	#print "pod limit exceeded",suma
	C,D=kernighan_lin_bisection(Ga)
	#print(C)
	#print(D)
        process(Ga,C)
	process(Ga,D)
     else:
	process_node(G,A)
	#print "the nodes from here-------------------------------------"
	#print(A)
	#print "pod limit not exceeded",suma
#--------------------------------Printing in file--------------------------------
         #f1=open('./data.txt', 'a+') 
         #f2=open('./app.txt', 'a+')
	 #for (u, data) in Ga.nodes.data('app'):
		#if data!=[]:
		#print u ,data
	 	#print>>f2, node_number_app,
		#for d in data:
			  #print>>f2,",",
			 # print>>f2,int(d),
		#print >>f2 ,"\n"
		#node_number_app=node_number_app+1
	#for (u, data) in Ga.nodes.data('data'):
		#if data!=[]:
		#print u ,data
	 	#print>>f1, node_number_data,
		#for d in data:
			  #print>>f1,",",
			  #print>>f1,int(d),
		#print >>f1 ,"\n"
		#node_number_data=node_number_data+1  
		#assert(node_number<=(pod_number+1)*int(k))   			
	#f2.close()
	#f1.close()
#--------------------------------Printing in file--------------------------------
	pod_number=pod_number+1
        node_number=pod_number*int(k)        
	#print "new pod and starting node",node_number,pod_number
	#assert(pod_number<k)
	#once we have the decision of the pod just decide about the node right away
        #process_node(G,A)
     return


def convertToList(string):
    L=[]
    #print(string)
    l=string.split(",")
    for i in l:
	if i=='':
		break
	else:
		L.append(int(i))
    return L

def fileparse():
    G=nx.MultiGraph()
    app=[]
    data=[]
    nodeSet=[]
    with open('./node.txt') as f:
       for line in f:
               field=line.split(";")
               app=convertToList(field[2][:-1])
               data=convertToList(field[3][:-1])
               G.add_node(int(field[0]), weight=int(field[1]),app=app,data=data) 
	       nodeSet.append(int(field[0]))
    f.close()
    with open('./edge.txt') as f:
    	for line in f:
        	field=line.split(",")
		G.add_weighted_edges_from([(int(field[0]),int(field[1]),int(field[2]))])
    f.close()
    return G,nodeSet


def partition():

     #parsing the files to read into the graph
     G,nodeSet=fileparse()
     #print(G.edges.data())
     #print(G.nodes.data())
     #---------------------------Processing----------------------------------------------
    # global pod_limit #= sys.argv[1]
    # global node_limit # = sys.argv[2]
     process(G,nodeSet)
#--------------------------------------------------------------------------------------------------partition ends here------------------------------------------------------------------------------- 


#-------------------------------------------------------------------------------------All the process are for mapping and its associate----------------------------------------------------------------

def editEdgeList(B,s,d):
	sum_weight=0
        for (src,dest,wt) in B.edges(data='weight'):
		if src==s and dest==d:
                        weight=B[src][dest]['weight']
                        sum_weight=sum_weight+weight
        return sum_weight


def find_reverse_edges(B,src,dest):
        sum_weight=0
        for (s,d,v) in B.edges(data='visited'):
                if src==s and dest==d and v!=1:
                        weight=B[s][d]['weight']
                        sum_weight=sum_weight+weight
        return sum_weight

def find_parallel_edges(B,src,dest):
        sum_weight=0
        for (s,d,v) in B.edges(data='visited'):
                if src==s and dest==d and v!=1:
                        weight=B[s][d]['weight']
                        sum_weight=sum_weight+weight
        return sum_weight


def findmax(B):
        maximum=0
	for (u,v,wt) in B.edges(data='weight'):
		if wt>=maximum:
			maximum=wt
			s=u
			d=v
	B[s][d]['visited']=1
	#print(maximum)
	return s,d,maximum	


def copy_content(B,src,dest,typ):
	#copying the data list to the source 
        data_src=B.nodes[src][typ]
        data_dest=B.nodes[dest][typ]
	#print data_dest
        for item in data_dest:
		if item not in data_src:
			data_src.append(item)
	return data_src

def exchange(edgeList,dest,src):
	for e  in edgeList:
		if e[0]==dest:
			e[0]=src
	for e in edgeList:
		if e[1]==dest:	
			e[1]=src	
	
def output_format(B,l):
	global node_id
#creating a mapping so that we can make the changes in the edgelist we can reuse the same function called exchange
	L2= [] 
	string=''
	L2.append(node_id)
	L2.append(l)
	f1=open('./node.txt', 'a+')
	string+=str(node_id)
	string+=';'
	node_id=node_id+1
	weight=B.nodes(data='io')
        app=B.nodes(data='app')
        data=B.nodes(data='data')
	
        for w in weight:
                if w[0]==l:
		       string+=str(w[1])
		       string+=';'
        for a in app:
                if a[0]==l:
                        for i in a[1:]:
					for j in i:
						string+=str(j)
						string+=','
					string+=';'
        for d in data:
                if d[0]==l:
                        for i in d[1:]:
                                        for j in i:
					       string+=str(j)
                                               string+=','
                                        string+=';'					       
	print>>f1,string
	f1.close()
	return L2

#this function is not in use
def find_match(B,src,dest,typ):
	#copying the data list to the source 
        data_src=B.nodes[src][typ]
	data_dest=B.nodes[dest][typ]
	#print "src = ",data_src
	#print "dest" , data_dest	
	for cL in copyList:
		copy_name=cL[1]
		if int(copy_name[1:]) in data_src:
			real_name=cL[0]
			if int(real_name[1:]) in data_dest:
				return true
	for cL in copyList:
		real_name=cL[0]
		if int(real_name[1:]) in data_src:
			copy_name=cL[1]
			if int(copy_name[1:]) in data_dest:
				return 1
	return 0

def merge():
  	#global node_cpu_limit #= 5 #Parameter
        global node_io_limit #=70 #Parameter
        global total_ssd_in_rack
	global total_node_fat_tree
	reverse_weight=0
	edgeList = []
	global copyList
	global app_count
	global chunk_count
	a='a'
	c='c'
	global baseAppRate #=100 #Parameter
	B = nx.DiGraph()
	p=app_count+1
	q=chunk_count+1
	for i in range (1,p):
		name=a+str(i)
		B.add_node(name,cpu=1,typ='app',io=0,valid=1,app=[i],data=[],exio=0)		# we have to fill in the CPU, we have assigned all 1
	#B.add_node('a1',cpu=1,typ='app',io=0,valid=1,app=[1],data=[],exio=0)
        #B.add_node('a2',cpu=1,typ='app',io=0,valid=1,app=[2],data=[],exio=0)
	#B.add_node('a3',cpu=2,typ='app',io=0,valid=1,app=[3],data=[],exio=0)
	#B.add_node('a4',cpu=1,typ='app',io=0,valid=1,app=[4],data=[],exio=0)
	
        for i in range (1,q):
                name=c+str(i)
		#print(name)
                B.add_node(name,cpu=0,typ='data',io=0,valid=1,app=[],data=[i],exio=0)         # io we will initialize 0 then update based on sum over edgelist
	#print(B.nodes.data())
        #B.add_node('c1',cpu=0,typ='data',io=45,valid=1,app=[],data=[1],exio=0)
        #B.add_node('c2',cpu=0,typ='data',io=60,valid=1,app=[],data=[2],exio=0)
        #B.add_node('c3',cpu=0,typ='data',io=27,valid=1,app=[],data=[3],exio=0)
        
        #for this we have to process the final file from mapping code
	fp=open('./final.txt', 'r')
        fa=open('./appfreq.txt', 'a+') # this file will capture the application frequency
        sumApp=0
	app=1
	#addition of edges to the graph
	for line in fp:
               field=line.split(",")
               app_number=field[0]
	       chunk_number=field[1]
	       #print field[0],field[1], field[2]
               app_name=a+app_number
	       chunk_name=c+chunk_number
	       #print app_name,chunk_name
               appToChunkIntensity=baseAppRate*float(field[2])   #*thisAPPRate
               if int(app_number)==app:
			sumApp=sumApp+appToChunkIntensity
	       else:
			stri=str(app)+","+str(sumApp)+","+str(float(sumApp/baseAppRate))
			print>>fa,stri
			#print>>fa,app,",",sumApp,",",float(sumApp/baseAppRate)
			app=int(app_number)
			sumApp=0
			sumApp=sumApp+appToChunkIntensity
               B.add_edge(app_name,chunk_name,weight=int(appToChunkIntensity),visited=0)   #not sure whether the weights will be in float or integer  //important
        stri=str(app)+","+str(sumApp)+","+str(float(sumApp/baseAppRate))
        print>>fa,stri
	#print>>fa,app,",",sumApp,",",float(sumApp/baseAppRate)
        fa.close()
	fp.close()
        #print(B.nodes.data())
	#print ("\n")
	#print(B.edges.data())

	#this part is related to copy creation in the begining
	srcList=[]        
	for (u,v,wt) in B.edges(data='weight'):
		 #print u,v,wt		
		 B.nodes[v]['io']+=wt #incoming  
		 B.nodes[u]['exio']+=wt  #outgoing  for app

        chunk_copy_number=chunk_count+1
	for (v,wt) in B.nodes(data='io'):
		if B.nodes[v]['io']> node_io_limit:
			srcList=[]   
			copy_cnt=int(math.ceil(B.nodes[v]['io']/node_io_limit))
			B.nodes[v]['io']=int(B.nodes[v]['io']/copy_cnt)
			for (s,d,w) in B.edges(data='weight'):
				if d == v:
					LK=[]
					#print w, "---", B.nodes[s]['exio']
					B.nodes[s]['exio']=B.nodes[s]['exio']-w
					w=int(w/copy_cnt)
					#print w , B.nodes[s]['exio']
					B.nodes[s]['exio']=B.nodes[s]['exio']+w
					#print B.nodes[s]['exio']
					#keep that src or application node number in a list that will help us to add new edges
					LK.append(s)
					LK.append(w)
					srcList.append(LK)
			#print srcList	
			i=0
			while i<copy_cnt-1:
				 LC=[]
				 number=int(v[1:])
				 name='c'+str(chunk_copy_number)
				 # keep the mapping here is a list before you increase the count
				 LC.append(number)
				 LC.append(chunk_copy_number)
				 copyList.append(LC)
				 B.add_node(name,cpu=0,typ='data',io=0,valid=1,app=[],data=[chunk_copy_number],exio=0)
      				 chunk_copy_number=chunk_copy_number+1
				 #add the edge between all the application in the src list and the new node 
				 for s in srcList:
					B.add_edge(s[0],name,weight=int(s[1]),visited=0)
					B.nodes[name]['io']+=int(s[1])
					#print name
				 i=i+1	
			#print "node---node", v , copy_cnt 	
			#print copyList
				

	#print(B.edges.data())	
        #for (u,v,wt) in B.edges(data='weight'):
	#	 B.remove_edge(u,v)

	#inside the loop try to calculate the sum of each chunk usage for example cumulative sum in an array

        #B.add_edge('a1', 'c1', weight=25 , visited=0 )
	#B.add_edge('a1', 'c2', weight=10 , visited=0 )
	#B.add_edge('a2', 'c2', weight=50 , visited=0 )
	#B.add_edge('a2', 'c3', weight=15 , visited=0 )
	#B.add_edge('a3', 'c3', weight=12 , visited=0 )
	#B.add_edge('a4', 'c1', weight=20 , visited=0 )
	
        #print(B.nodes.data())
       	#print(B.edges.data())
        
	while B.number_of_edges() !=0 :
		src,dest,wt=findmax(B)
		#print(src,dest,wt)
		reverse_weight=0
	        src_type= B.nodes[src]['typ']
		dest_type= B.nodes[dest]['typ']
                #print(B.edges.data())
#------------------------------------------------treatment for data---------------------------------------
                if dest_type=='data':
			parallel_weight=find_parallel_edges(B,src,dest)
			assert(parallel_weight<=wt)
			#if (parallel_weight>wt):
				#print "parallel weight",parallel_weight,"weight",wt
			#print "checking the avail before atod merge",B.nodes[src]['io']+wt
 			if B.nodes[src]['io']+wt+B.nodes[dest]['io']<node_io_limit and (B.nodes[src]['exio']-wt)+B.nodes[dest]['exio']<node_io_limit:
				#increase the io usage at the source as we are merging
				B.nodes[src]['io']=B.nodes[src]['io']+B.nodes[dest]['io']
				#copying the data list to the source 
				data_src=copy_content(B,src,dest,'data')
				B.nodes[src]['data']=data_src
				#print(B.nodes[src]['data'])
				#decrease the total io demand of the data chunk node
				B.nodes[src]['exio']=B.nodes[src]['exio']-wt;
				#update external io of the src node as we merge the dest data node
				B.nodes[src]['exio']=B.nodes[src]['exio']+B.nodes[dest]['exio']
				#print(B.nodes[src]['exio'])
				#mark the destination node as invalid since it is now merged to src application node
                                B.nodes[dest]['valid']=0
				#print "weight:",wt,"reverse",reverse_weight,"exio src",B.nodes[src]['exio'],"exio_dest",B.nodes[dest]['exio']   important
				B.remove_edge(src,dest) # this will remove all the edges that are between this src and destination, so make sure
				#exchange the dest in the edgelist and wherever dest found change it to src
                		for (u,v,wt) in B.edges(data='weight'):   # important
                        		if v==dest:	
                                		B.remove_edge(u,v) # this will remove all the edges that are between this src and destination, so make sure
						#print (u,v,wt,src,dest)
                                		B.add_edge(u,src, weight=wt , visited=0 )
                                #print(B.edges.data())
#$$$ commented the next four lines for real traces
                		#for (s,d,w) in B.edges(data='weight'):
                        	#	tot_weight=editEdgeList(B,s,d) #calculate the total weight for a (s,d) pair     
                        	#	B.remove_edge(s,d) #remove all such edges or parallel edges
                        	#	B.add_edge(s, d, weight=tot_weight , visited=0 ) # then insert only one edge with that budget
					#after every successful merge try to change the entries in the edgelist
				exchange(edgeList,dest,src)
                                #print(B.edges.data())

					
  			else:
				L=[]
                                L.append(src)
                                L.append(dest)
                                L.append(wt)
                                edgeList.append (L)
				B.remove_edge(src,dest) # this will remove all the edges that are between this src and destination, so make sure
				#print ("we could not place the above relationship")
		if dest_type=='app' :
			# before this if condition we have to check for reverse edge between these two nodes						
			#print("cpu")
			#print(B.nodes[src]['cpu']+B.nodes[dest]['cpu'])
			#if there is a self loop then safely delete the edge and donot do anything
			if (src==dest):
				 B.remove_edge(src,dest)
				# print "**********************************",src
				 continue	
			reverse_weight=find_reverse_edges(B,dest,src)
			parallel_weight=find_parallel_edges(B,src,dest)
			assert(parallel_weight<=wt)
			#if(parallel_weight>wt):
			#	print "parallel weight",parallel_weight,"weight",wt
			#print "checking the avail before atoa merge srcio", B.nodes[src]['io'], "weight",wt,"destination_io",B.nodes[dest]['io'],"reverse weight",reverse_weight
                        if B.nodes[src]['io']+B.nodes[dest]['io']<node_io_limit and ((B.nodes[src]['exio']-wt)+(B.nodes[dest]['exio']-reverse_weight))<node_io_limit: #and B.nodes[src]['cpu']+B.nodes[dest]['cpu']<node_cpu_limit:
                               # B.nodes[src]['cpu']=B.nodes[src]['cpu']+B.nodes[dest]['cpu'] #increase cpu at src
                                B.nodes[src]['io']=B.nodes[src]['io']+B.nodes[dest]['io']
				#decrease the cpu at destination to 0
				#B.nodes[dest]['cpu']=0 

				#update the external io
				externaliosum= B.nodes[src]['exio']+ B.nodes[dest]['exio']
				#print "external io sum of two nodes",(externaliosum)
				externaliosum=externaliosum-(wt+reverse_weight)
				#print "weight:",wt,"reverse",reverse_weight,"exio src",B.nodes[src]['exio'],"exio_dest",B.nodes[dest]['exio']
				#print "external io sum final",(externaliosum)
				B.nodes[src]['exio']=externaliosum

				#copy the data and the application list from the dest to the src list
				data_src=copy_content(B,src,dest,'data')
                		B.nodes[src]['data']=data_src
				app_src=copy_content(B,src,dest,'app')
				B.nodes[src]['app']=app_src
                                #print(B.nodes[src]['app'])
				#print(B.nodes[src]['data'])

				#mark the destination node as invalid since it is now merged to src application node
				B.nodes[dest]['valid']=0
				B.remove_edge(src,dest) # this will remove all the edges that are between this src and destination, so make sure
                                #remove all the reversed edge
				if reverse_weight>0:
					B.remove_edge(dest,src)
				#exchange the dest in the edgelist and wherever dest found change it to src
				for (u,v,wt) in B.edges(data='weight'):
                        		if u==dest: # its a bit different case for app-to-app
                                		B.remove_edge(u,v) # this will remove all the edges that are between this src and destination, so make sure
          	                		B.add_edge(src, v, weight=wt , visited=0 )
       				#exchange the dest in the edgelist and wherever dest found change it to src
                                for (u,v,wt) in B.edges(data='weight'):
                        		if v==dest:
                                		B.remove_edge(u,v) # this will remove all the edges that are between this src and destination, so make sure
                                		B.add_edge(u, src, weight=wt , visited=0 )
#$$ commented the next few lines for real trace
                		#for (s,d,w) in B.edges(data='weight'):
                        	#	tot_weight=editEdgeList(B,s,d) #calculate the total weight for a (s,d) pair     
                        	#	B.remove_edge(s,d) #remove all such edges or parallel edges
                        	#	B.add_edge(s, d, weight=tot_weight , visited=0 ) # then insert only one edge with that budget
				#after every successful merge we try to change the entries in the edgelist
				exchange(edgeList,dest,src)
			else:
				L=[]
				L.append(src)
				L.append(dest)
				L.append(wt)
				edgeList.append (L)
				B.remove_edge(src,dest) # this will remove all the edges that are between this src and destination, so make sure
                        	#print("we could not merge")
		
				
    	#B.remove_edge(src,dest) # this will remove all the edges that are between this src and destination, so make sure
	#print("printing the edgelist")       
	#print(edgeList)
	#print(B.nodes.data())
	#print only the valid data
	# 3jobs (1) what to do with the rest of the nodes which we cant assin (2) what to do with isoldated node with 0 wweigth (3) how to process the input and the ouput(op is more important) 
	#important onwardb this	
	exio=B.nodes(data='exio')
	#print exio
	L1= []

	#need valid and node number
	for (s,v) in B.nodes(data='valid'):
                if v==1:
			L1.append(s)
        #print (L1) 
	# if a node is valid yet the external io is 0 then we have to create an edge with cost 0
	#for ex in exio :
		#if ex[0] in L1 and ex[1]==0:
			#for item in L1:
				#if item !=ex[0]:
                                       # P=[]
 	                               # P.append(ex[0]) 
					#P.append(item) 
					#P.append(1) # then insert only one edge with that budget
					#edgeList.append(P)

	
	L2 = []
        for (s,wt) in B.nodes(data='io'):
		if s in L1:
                	P=[]
                	P.append(s)
                	P.append(wt)
                	L2.append(P)
        #print(L2)
        LL2=sorted(L2, key=itemgetter(1,0), reverse=True)
        #print(LL2)

	#this part in bin-packing
	max_node=int(total_node_fat_tree*total_ssd_in_rack)
	actualNode=0
	w = int(total_node_fat_tree*total_ssd_in_rack)
	lastsrc=-1
	LList=[]
	LList2=[]
	bigList=[]
	maxval=budget
	Matrix = [0 for x in range(w)]
	#print Matrix 
	for d in LL2:
		for node,weight in zip(*[iter(d)]*2):			
			if maxval>=weight:
		                maxval=maxval-weight
				Matrix[actualNode]=Matrix[actualNode]+weight 
				#print(node)	
			        if lastsrc==-1:
					lastsrc=node #assign the first virtual node to real node 
					LList2.append(node)
				if lastsrc!=-1:
					data_src=copy_content(B,lastsrc,node,'data')
        				B.nodes[lastsrc]['data']=data_src
					app_src=copy_content(B,lastsrc,node,'app')
					B.nodes[lastsrc]['app']=app_src
					#else : 
						#print "match found----"
				LList.append(node)
			else:
				#print(node)
				bigList.append(LList)
				B.nodes[lastsrc]['io']=Matrix[actualNode]
				lastsrc=-1
				LList=[]
				maxval=budget				
				if(actualNode+1<max_node):
					actualNode=actualNode+1
				else :
				   	print("node limit exceeded")
				lastsrc=node
				LList2.append(node)
				LList.append(node)				
		                maxval=maxval-weight
				Matrix[actualNode]=Matrix[actualNode]+weight 
	
	B.nodes[lastsrc]['io']=Matrix[actualNode]
	bigList.append(LList)
	#print(Matrix)
        #print(bigList)	
	#print LList2, "there we go"
	for ll2 in LList2:
		data_src=B.nodes[ll2]['data']
		#print "chunk: ",data_src
        	app_src=B.nodes[ll2]['app']
		#print "application: ",app_src
		io = B.nodes[ll2]['io']
		#print "total node weight: ",io
	
	global node_id
	LM=[]	
	P1=[]


	for l in bigList:
		for p in l:
			LM.append(node_id)
			LM.append(p)
			P1.append(LM)
			LM=[]
		node_id=node_id+1

	#creating the nodelist
	f1=open('./node.txt', 'a+')
	for ll2 in LList2:
		string=''
		for e in P1:
			if e[1]==ll2:
				#print e[0],e[1],ll2
				string+=str(e[0])
				string+=';'
				io = B.nodes[ll2]['io']
				string+=str(io)
				string+=';'
				app_src=B.nodes[ll2]['app']
				for ip in app_src:
					string+=str(ip)
					string+=','
				string+=';'
				data_src=B.nodes[ll2]['data']
				for ip in data_src:
					string+=str(ip)
					string+=','
				string+=';'
				print>>f1,string
	f1.close()




	
	Matrix2 = [[0 for x in range(1,node_id)] for y in range(1, node_id)]
	#print(Matrix2)
	nodeid=node_id-1			
	#L3=[]						
	#for l in L1:
		#L4=output_format(B,l)
		#L3.append(L4)
	#print(L3)			
	# creating the final edge list and assigning to file
	for item in P1:
		exchange(edgeList,item[1],item[0])
	#print(edgeList)
	

	
	for e in edgeList:
		#print e
		if int(e[0])<int(e[1]):
			#print("normal")
			Matrix2[(int(e[0])-1)][(int(e[1])-1)]=Matrix2[(int(e[0])-1)][(int(e[1])-1)]+int(e[2])
			
		else:
			#print("reverse")
			Matrix2[(int(e[1])-1)][(int(e[0])-1)]=Matrix2[(int(e[1])-1)][(int(e[0])-1)]+int(e[2])
	#print(Matrix2)
		
	LLSE=[]
	S=[]
	for s in range(1,(nodeid)):
			t=s+1
			while int(t)<=int(nodeid):
				#print s,t
				#print s-1, t-1
				S.append(s)
				S.append(t)	
				val=Matrix2[(s-1)][(t-1)]
				#print val
				S.append(val)
				LLSE.append(S)
				S=[]
				t=t+1
	#print(LLSE)	

						
	f2=open('./edge.txt', 'a+')	        
	
	for e in LLSE:
		edgestr=''
		edgestr+=str(e[0])
		edgestr+=','
		edgestr+=str(e[1])
		edgestr+=','
		edgestr+=str(e[2])
		edgestr+=','
		print>>f2,edgestr
        f2.close()
	
	#for edge in edgeList:
	#	edgestr=''
	#	for e in edge:
	#		edgestr+=str(e)
	#		edgestr+=','
	#	print>>f2,edgestr
        #f2.close()
	



#-------------------------------------------------------------------------------Mapping--------------------------------------------------
def perm(a, k=0):
   if k == len(a):
      print a
   else:
      for i in xrange(k, len(a)):
         a[k], a[i] = a[i] ,a[k]
         perm(a, k+1)
         a[k], a[i] = a[i], a[k]


def hs_sum(n,alpha):
	summ=0.0
	n=n+1
  	for i in range(1,n):
  		j=1.0/float(i)
    		val=pow(j,alpha)
    		summ+=val
	return summ

def zipff(alpha,n,x):
 	numerator=1.0
 	denominator1 = hs_sum(n,alpha)
 	denominator2=pow(x,alpha)
 	denominator=denominator1*denominator2
 	expected_val=numerator/denominator
	return expected_val



def count(apps,x):
	count=0
	for ap in apps:
		if x in ap:
			count=count+1
	return count

def assign():
	global app_count
	global chunk_count
	global loop_count
        global alpha
	q=chunk_count+1
        apps = [[] for i in range(0, app_count)]
	if(seed):
        	random.seed(seed)
        for i in range(loop_count):
		j=random.randint(0,app_count-1) # since the first list in a list of lists is indexed as 0 but app no starts from 1
		#print "the random number generated is", j	
		for k in range(1,q):
				while k in apps[j]:
					j=(j+1)
					if(j>=app_count):
						j=0
				apps[j].append(k)
				j=(j+1)
                                if(j>=app_count):
					j=0
		     
                                	
       
	#print(apps)
	#a=2.0
	#result=numpy.random.zipf(a,5)
	#print "result is",  result
	#alpha=1 please set it in global param
	#n=3
	#x=1
	#expected=zipff(alpha,n,x)
	f2=open('./final.txt', 'a+')
	#print(expected)
	
	i=1 # this represents the application number
	for ap in apps:
		ap.sort()
		length=(len(ap))+1
		#P=[]
		for x in ap:# range(1,length):
			cnt=count(apps,x)	

			z=zipff(alpha,chunk_count,int(x))
			#P.append(y)  # this is of no use
			#j=x-1
			#y=z/cnt  
			y=format((z/cnt),'.12f') #loop_count # this needs to be improvised for sophisticated placement
			string=str(i)+","+str(x)+","+str(y)
			#string=str(i)+","+str(ap[j])+","+str(y)
			print>>f2,string
			#print>>f2,i,",",ap[j],",",y
		#print(P)
		i=i+1
	f2.close()
	#for app in apps:
		#perm(app)

def assign_trace():
        sum_weight=0
        fp=open('./chunk_application_intensity.csv','r')
        for line in fp:
               field=line.split(",")
              # app_number=field[0]
              # chunk_number=field[1]
               sum_weight = sum_weight + float(field[2])   #*thisAPPRate
        #print (sum_weight)
        fp.close()
        fp=open('./chunk_application_intensity.csv','r')
        fa=open('./final.txt','w')
        for line in fp:
               field=line.split(",")
               app_number=field[0]
               chunk_number=field[1]
               string=app_number+","+chunk_number+","+str(format((float(field[2])/sum_weight),'.12f'))
               print>>fa,string
        fa.close()
        fp.close()
        return


def getKey(item):
	return item[0]
def main():
	global LL
	global LP
	trace=int(sys.argv[1])
	clean_all()
        if trace==1:
		assign_trace()
	else:
		assign()
        merge()	
        partition()
	#LP=sorted(LL, key=itemgetter(1,0), reverse=True)
	#print(LP)
	#binPacking()
if __name__ == "__main__":
     main()

