#!/usr/bin/env python
# coding: utf-8

# # Setup

# ## Import libraries

# ### Data processing

# In[119]:


from datetime import datetime
import pandas as pd
import numpy as np
import math
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import multivariate_normal, pearsonr
import scipy.integrate as integrate
from sklearn.neighbors import KernelDensity
from pathlib import Path
import networkx as nx
import json


# ### Visualization

# In[2]:


import matplotlib.pyplot as plt

#Seaborn
import seaborn as sns
sns.set_style("whitegrid")

#Plotly
import plotly
import plotly.tools as tls
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

# If you're using this code locally:
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# If you're copying this into a jupyter notebook, add:
init_notebook_mode(connected=True)


# ## Parameter settings

# In[3]:


chart = True # boolean for whether to display images while running computation
debug = True # boolean for whether to print updates to the console while running
output = True # boolean for whether to output json and pngs to files
resolution = 150 # int for resolution of output plots
discrete_threshold = 5 # number of responses below which numeric responses are considered discrete
compare_all = True # boolean; if comparing two lists of the same length, fill in list1 and list2 accordingly
#list1, list2 = [],[]
cd = 'demo_widsdatathon2020/'

if output:
    Path(cd+'output/').mkdir(parents=True, exist_ok=True)
    Path(cd+'output/charts').mkdir(parents=True, exist_ok=True)
    Path(cd+'output/json').mkdir(parents=True, exist_ok=True)


# ## Import Data

# In[4]:


"""
df = pd.read_csv('demo_eusocial/eusocial.csv',low_memory=False).drop(columns=['idno']).sample(5000)
compare_all = True
for i in list(df.columns):
    try: np.ma.fix_invalid(df[i])
    except: pass
"""


# In[5]:


"""
df = pd.read_csv('demo_cost-of-living/cost-of-living.csv').T.drop(index=['Unnamed: 0'])
compare_all = True
"""


# In[6]:


"""
df = pd.read_csv('demo_datasaurus/DatasaurusDozen-wide.tsv',sep='\t')
compare_all = False
list1 = []
list2 = []
for i in list(df.columns):
    if '_x' in i: list1.append(i)
    else: list2.append(i)
"""


# In[7]:


df = pd.read_csv('demo_widsdatathon2020/training_v2.csv').sample(1000)
compare_all = True
ids = []
for i in list(df.columns):
    try: np.ma.fix_invalid(df[i])
    except: pass
    if '_id' in i:
        ids.append(i)
df = df.drop(columns=ids)


# In[8]:


df.shape


# # Helper functions

# ## Identify feature type

# In[9]:


# Get a list of all response types
response_list = pd.DataFrame(columns=['responses','types'], index=list(df.columns))
response_list['responses']=[list(df[col].value_counts().index) for col in df.columns]

# Delete columns from the dataframe that only have one response
response_list['only_one_r'] = [(len(r)<2) for r in response_list['responses']]
only_one_r = list(response_list[response_list['only_one_r']==True].index)
df = df.drop(columns=only_one_r)
response_list = response_list.drop(index=only_one_r)


# In[10]:


def get_types(U):
  types = {'floats':0,'strings':0,'nulls': 0}
  for i in response_list['responses'][U]:
    try:
      val = float(i)
      if math.isnan(val)==False:
        #print("Value",i," is a float")
        types['floats']+=1
      else:
        #print("Value",i," is null")
        types['nulls']+=1
    except ValueError:
      try:
        val = str(i)
        #print("Value",i,"is a string")
        types['strings']+=1
      except:
        print('Error: Unexpected value',i,'for feature',U)
  if ((types['floats']>0) & (types['strings']>0)):
    print('Hey! Column',U,'contains floats AND strings')
  return types


# In[11]:


response_list['types']=[get_types(col) for col in df.columns]


# In[12]:


response_list['string']=[t['strings']>0 for t in response_list['types']]
response_list['float']=[t['floats']>0 for t in response_list['types']]


# In[13]:


# Classify features as discrete (fewer than {discrete_threshold} responses, or contains strings) or continuous (more than 15)
response_list['class']=['d' if ((len(r) < discrete_threshold) or (t['strings']>0)) else 'c' for r,t in zip(response_list['responses'],response_list['types'])]

# Store these groups in a list
discrete = list(response_list[response_list['class']=='d'].index)
continuous = list(response_list[response_list['class']=='c'].index)


# In[14]:


# Format the data as a string or a float
for i in list(response_list.index):
    V = []
    if (response_list['string'][i]==True) or (response_list['class'][i]=='d'):
        V=[str(v) for v in df[i]]
        df[i]=V
    elif response_list['float'][i]==True:
        V = df[i]
        V=[float(v) for v in df[i]]
        #V=np.ma.fix_invalid(df[i])
        df[i]=V
    else: print('Error formatting column ',i)


# In[15]:


df = df.replace(np.nan, None)
df = df.replace('nan', None)


# ## Data structures

# In[16]:


def sparsify(series):
  ''' For discrete values: takes a column name and returns a sparse matrix (0 or 1) with a column for each unique response '''
  m=pd.DataFrame(columns=list(series.unique()))
  for i in list(series.unique()):
    m[i]=[int(x==i) for x in series]
  return m


# In[17]:


def compute_bandwidth(X):
  ''' Takes a column name and computes suggested gaussian bandwidth with the formula: 1.06*var(n^-0.2) '''
  var = np.var(df[X])
  n = len(df[X].notnull())
  b = 1.06*var*(n**(-0.2))
  return b


# # Main Functions

# ## Visualization

# ### Discrete-Discrete Confusion Matrices

# In[18]:


def DD_viz(df):
    
  ''' Takes two discrete feature names and generates a heatmap '''

  U=list(df.columns)[0]
  V=list(df.columns)[1]

  i_range = list(df[U].unique())
  j_range = list(df[V].unique())
  s = pd.DataFrame(columns=i_range,index=j_range)
  for i in i_range:
    for j in j_range:
      s[i][j]=df[(df[U]==i) & (df[V]==j)].filter([U,V],axis=1).shape[0]
      mutual_support=s.sum().sum()
  s = s.astype(int)
  plt.clf()
  plt.figure(dpi=resolution)
  sns.heatmap(s, annot=True, cmap="Blues", cbar=False, linewidths=1)
  plt.xlabel(U)
  plt.ylabel(V)
  return 


# In[19]:


try: DD_viz(df.filter([discrete[0],discrete[1]]).dropna(how='any'))
except: pass


# ### Discrete-Continuous Violin Plots

# In[20]:


def DC_viz(df):

  ''' Takes one continuous and one discrete feature name and generates a Violin Plot '''

  U=list(df.columns)[0]
  V=list(df.columns)[1]
  
  if (U in continuous):
    D = V
    C = U
  else:
    D = U
    C = V
    
  sns.violinplot(df[D], df[C])
  if (len(df[D]) < 500): sns.swarmplot(x=df[D], y=df[C], edgecolor="white", linewidth=1) # Only show a swarm plot if there are fewer than 500 data points
  plt.xlabel(D)
  plt.ylabel(C)
    
  if output:
    plt.savefig(str(cd+'output/charts/'+U+'_'+V+'.png'), dpi=resolution)
    df.to_json(str(cd+'output/json/'+U+'_'+V+'.json'))

  return


# In[21]:


try: DC_viz(df.filter([continuous[0],discrete[0]]))
except: print('Error building discrete-continuous viz')


# ### Continuous-Continuous KDE Plots

# In[22]:


def CC_viz(df):

  ''' Takes two continuous feature names and generates a 2D Kernel Density Plot '''

  U=list(df.columns)[0]
  V=list(df.columns)[1]

  sns.kdeplot(df[U], df[V], color='blue',shade=True, alpha=0.3, shade_lowest=False)
  if (len(df[U]) < 500): sns.scatterplot(x=df[U], y=df[V], color='blue', alpha=0.5, linewidth=0) # Only show a scatter plot if there are fewer than 500 data points
  
  plt.xlabel(U)
  plt.ylabel(V)
    
  if output:
    plt.savefig(str(cd+'output/charts/'+U+'_'+V+'.png'), dpi=resolution)
    df.to_json(str(cd+'output/json/'+U+'_'+V+'.json'))

  return 


# In[23]:


sns.distplot(df[continuous[0]].dropna(how='any'))


# In[24]:


try:CC_viz(df.filter([continuous[0],continuous[1]]).dropna(how='any'))
except: print('Error building continuous-continuous viz')


# ### Matrix Heatmap

# In[25]:


def matrix_viz(matrix):
   plt.clf()
   plt.figure(dpi=70,figsize=(10,8))
   sns.heatmap(matrix.fillna(0))
   plt.show()
   return()


# ### Visualization Function Router

# In[26]:


def viz(U,V):

  ''' Generate a visualization based on feature types '''

  plt.clf()
  plt.figure(dpi=resolution)

  pairdf = df.filter([U,V]).dropna(how='any')
  
  # If both features are discrete:
  if ((U in discrete) and (V in discrete)):
    viz = DD_viz(pairdf)
  # If both features are continuous:
  elif ((U in continuous) and (V in continuous)):
    viz = CC_viz(pairdf)
  # If one feature is continuous and one feature is discrete:
  elif (((U in continuous) and (V in discrete)) or ((U in discrete) and (V in continuous))):
    viz = DC_viz(pairdf)
  else:
    viz = print('Error on features',U,'and',V)
    
  if chart:
    try: plt.show()
    except: pass
    
  return viz


# ## Mutual Information

# ### Discrete-Discrete

# In[27]:


def DD_mi(df):

  ''' Takes two discrete feature names and calculates normalized mutual information (dividing mutual information by maximum possible) '''

  U=list(df.columns)[0]
  V=list(df.columns)[1]

  if debug: print('Calculating discrete-discrete normalized MI for',U,'and',V)
  min_response_count = min(len(list(df[U].unique())),len(list(df[V].unique())))
  max_mi = np.log2(min_response_count)
  if U == V:
      mi = max_mi
  else:
      i_range = list(df[U].unique())
      j_range = list(df[V].unique())
      # We use 's' to denote a matrix of support for each i,j
      s = pd.DataFrame(columns=i_range,index=j_range)
      for i in i_range:
        for j in j_range:
          s[i][j]=df[(df[U]==i) & (df[V]==j)].filter([U,V],axis=1).shape[0]
          mutual_support=s.sum().sum()
      s = s.astype(int)
      pmi = s.copy()
      l = []
      # If these features are never both answered, or if either feature only has one possible response:
      if (mutual_support <= 0 or len(i_range)<=1 or len(j_range)<=1):
        # The whole pointwise mutual information matrix should be 0
        pmi.fillna(0,inplace=True)
      else:
        for i in i_range:
          for j in j_range:
            joint_support=s[i][j]
            joint_probability = joint_support/mutual_support
            marginal_probability_i=s.sum(axis=0)[i]/s.sum().sum()
            marginal_probability_j=s.sum(axis=1)[j]/s.sum().sum()
            if joint_probability !=0:
              pmi[i][j]=np.log2(joint_probability/(marginal_probability_i*marginal_probability_j))
              # Store all PMI (pointwise mutual information) in a list
              l.append(pmi[i][j]*joint_probability)
        # Sum the list of all pointwise mutual information
        mi = sum(l)
        if (chart or output): viz(U,V)
  if max_mi==0:
      nmi = 0
  else:
      nmi = mi/max_mi
  return nmi


# In[28]:


# Test it out:
try: DD_mi(df.filter([discrete[0],discrete[1]]).dropna(how='any'))
except: print('Error calculating MI for two discrete features')


# ### Discrete-Continuous
# 
# This uses SciKit's [mutual_info_regression](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html) which calculates mutual information using the narest neighbor entropy approach described in [*B. C. Ross “Mutual Information between Discrete and Continuous Data Sets”. PLoS ONE 9(2), 2014.*](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0087357)
# 
# This requires us to sparsify the discrete matrix by response

# In[29]:


def DC_mi(df):

  ''' Takes one discrete and one continuous feature name and, using a sparsified matrix of the discrete responses, returns mutual information score '''

  U=list(df.columns)[0]
  V=list(df.columns)[1]

  if debug: print('Calculating discrete-continious MI for',U,'and',V)
  if (U in continuous):
    D = V
    C = U
  else:
    D = U
    C = V
  if debug: print('Discrete:',D,'Continuous:',C)
  responses = list(df[D].unique())
  if debug: print('responses = ',responses)
  pmi = list(mutual_info_regression(sparsify(df[D]), df[C], discrete_features=True))
  if debug: print('pmi list = ',pmi)
  l = []
  for i in list(range(0,(len(responses)))):
    conditional_probability = df[df[D]==responses[i]].shape[0]/len(df[C])
    l.append(pmi[i]*conditional_probability)
  mi = sum(l)
  return mi


# In[30]:


# Test it out:
try: DC_mi(df.filter([discrete[0],continuous[0]],how='any'))
except: pass


# ### Continuous-Continuous

# In[31]:


def CC_mi(df):

  ''' Takes two continuous feature names and calculates mutual information using SciKit's mutual_info_regression function '''

  U=list(df.columns)[0]
  V=list(df.columns)[1]

  mi = mutual_info_regression(df.filter([U]),df[V])[0]
  return mi


# In[32]:


# Test it out:
try: CC_mi(df.filter([continuous[0],continuous[1]]))
except: pass


# #### Comparison: Correlation

# In[33]:


def CC_corr(df):

  ''' Takes two continuous feature names and calculates pearson correlation '''

  U=list(df.columns)[0]
  V=list(df.columns)[1]

  corr = pearsonr(df[U], df[V])[0]
  return corr


# In[34]:


# Test it out:
try: CC_corr(df.filter([continuous[0],continuous[1]]))
except: pass


# ### Mutual Information Function Router

# In[35]:


def calc_pairtype(U,V):

  ''' Takes two feature names and returns the pair type ('DD': discrete/discrete, 'DC': discrete/continuous, or 'CC': continuous/continuous) '''

  if debug: print('Finding pair type for "',U,'" and "',V,'"')
  # If both features are discrete:
  if ((U in discrete) and (V in discrete)):
    pair_type = 'DD'
    if debug: print('"',U,'" and "',V,'" are data pair type',pair_type)
  # If both features are continuous:
  elif ((U in continuous) and (V in continuous)):
    pair_type = 'CC'
    if debug: print('"',U,'" and "',V,'" are data pair type',pair_type)
  # If one feature is continuous and one feature is discrete:
  elif ((U in continuous) and (V in discrete)) or ((U in discrete) and (V in continuous)):
    pair_type = 'DC'
    if debug: print('"',U,'" and "',V,'" are data pair type',pair_type)
  else:
    pair_type = 'Err'
    print('Error on',U,'and',V)
  return pair_type


# In[36]:


def calc_mi(U,V):

  ''' Takes two feature names and determines which mutual information method to use; returns calculated mutual information score '''

  try:
      pairdf = df.filter([U,V]).dropna(how='any')

      if pairdf.shape[0]<1:
        return 0

      if debug:
        print('Calculating mutual information for',U,'(',list(df.columns).index(U),'of',len(list(df.columns)),')',V,'(',list(df.columns).index(V),'of',len(list(df.columns)),')')
      mi_start_time = datetime.now()
      if (U==V):
        return 1
      else:
        pair_type = calc_pairtype(U,V)
        # If both features are discrete:
        if (pair_type=='DD'):
          mi = DD_mi(pairdf)
        # If both features are continuous:
        elif (pair_type=='CC'):
          mi = CC_mi(pairdf)
        # If one feature is continuous and one feature is discrete:
        elif (pair_type=='DC'):
          mi = DC_mi(pairdf)
        else:
          mi = 0
        if (chart or output):
            viz(U,V)
        if debug:
            print('MI:',mi)
        if debug: print('Elapsed time:',datetime.now() - mi_start_time)
      return mi
  except: return 0


# ## Feature Network

# In[37]:


def stack(matrix,chart=False):
  ''' For undirected matrices: Takes a matrix and returns a dataframe with columns [x,y,v] corresponding to [source,target,value] '''
  s = pd.DataFrame(matrix.mask(np.triu(np.ones(matrix.shape)).astype(bool)).stack()).reset_index().rename(columns={'level_0':'x','level_1':'y',0:'v'})
  s = s[s['x']!=s['y']]
  s = s[s['v']>0]
  s = s.sort_values(by='v',ascending=False)
  #sns.distplot(s['v'],kde=False)
  return s


# In[38]:


def matrixify(df):
  ''' Takes a dataframe with columns [source,target,value] and returns a matrix where {index:source, columns:target, values:values} '''
  m = df.pivot(index=list(df.columns)[0], columns=list(df.columns)[1], values=list(df.columns)[2])
  return m


# In[39]:


def run_calc(features):
  start_time = datetime.now()

  if compare_all: matrix = pd.DataFrame(1, columns=features,index=features)
  else: matrix = pd.DataFrame(1, columns=list1,index=list2)
        
  s = stack(matrix)

  s['v'] = [calc_mi(x,y) for x,y in zip(s['x'],s['y'])]
  if debug:
    print('Elapsed time:',datetime.now() - start_time)
    print('Calcuated mutual information for',len(features),'columns across',df.shape[0],'records')
  return s


# In[40]:


output=False
chart=False
s = run_calc(list(df.columns))


# In[41]:


s.to_csv(cd+'results.csv',index=False)


# In[42]:


s = pd.read_csv(cd+'results.csv')


# In[43]:


sorted_stack = s.sort_values(by='v',ascending=False)
sorted_stack = sorted_stack[sorted_stack['v']<1]


# In[44]:


e = pd.DataFrame(columns=['mi_threshold','edge_count','components'])


# In[86]:


for i in np.arange(np.round(sorted_stack['v'].min(),2), np.round(sorted_stack['v'].max(),2), 0.01):
    
    s = sorted_stack[sorted_stack['v']>i]
    
    G = nx.Graph()
    G.add_nodes_from(list(dict.fromkeys((list(s['x'].unique())+list(s['y'].unique())))))
    G.add_edges_from(list(zip(s['x'],s['y'])))
    
    e = e.append({'mi_threshold': i, 'edge_count': (sorted_stack['v']>i).sum(), 'components':nx.number_connected_components(G)},ignore_index=True)


# In[87]:


sns.lineplot(e['mi_threshold'],e['edge_count'])


# In[88]:


sns.lineplot(e['mi_threshold'],e['components'])


# In[89]:


max_component_threshold = e[e['components']==max(e['components'])].max()['mi_threshold']


# In[90]:


thresh_stack = sorted_stack[sorted_stack['v']>max_component_threshold]
thresh_stack = thresh_stack.rename(columns={'x':'src','y':'target','v':'weight'})
thresh_stack['viztype']=[calc_pairtype(x,y) for x,y in zip(thresh_stack['src'],thresh_stack['target'])]
thresh_stack


# In[95]:


# Create a networkx graph from the list of pairs
G=nx.from_pandas_edgelist(thresh_stack, 'src', 'target', ['weight'])


# In[98]:


dict(G['d1_lactate_min'])


# In[115]:


nodelist = {}
for n in list(dict.fromkeys((list(s['x'].unique())+list(s['y'].unique())))):
    nodelist[n]={'type':'continuous' if (response_list['class'][n])=='c' else 'discrete','neighbors':dict(G[n])}


# In[116]:


nodelist


# In[126]:


json_out = {}
json_out['edges']=(thresh_stack).to_dict(orient='records')
json_out['nodes']=nodelist

with open(str(cd+'output/graph.json'), 'w') as json_file:
  json.dump(json_out, json_file)


# In[52]:


output=True
chart=True

for i,row in thresh_stack.iterrows():
    viz(row['src'],row['target'])


# In[68]:


def calculate_positions(thresh_stack):

  # Generate position data for each node:
  #pos=layout(G)
  # if weighted:
  pos=nx.kamada_kawai_layout(G, weight='weight')
      
  # Save x, y locations of each edge
  edge_x = []
  edge_y = []

  # Calculate x,y positions of an edge's 'start' (x0,y0) and 'end' (x1,y1) points
  for edge in G.edges():
      x0, y0 = pos[edge[0]]
      x1, y1 = pos[edge[1]]
      edge_x.append(x0)
      edge_x.append(x1)
      edge_y.append(y0)
      edge_y.append(y1)

  # Bundle it all up in a dict:
  edges = dict(x=edge_x,y=edge_y)

  # Save x, y locations of each node
  node_x = []
  node_y = []

  # Save node stats for annotation
  node_name = []
  node_adjacencies = []
  node_centralities = []

  # Calculate x,y positions of nodes
  for node in G.nodes():
      node_name.append(node)# Save node names
      x, y = pos[node]
      node_x.append(x)
      node_y.append(y)

  for node, adjacencies in enumerate(G.adjacency()):
      node_adjacencies.append(len(adjacencies[1]))

  for n in G.nodes():
      node_centralities.append(nx.degree_centrality(G)[n])

  # Bundle it all up in a dict:
  nodes = dict(x=node_x,y=node_y,name=node_name,adjacencies=node_adjacencies,centralities=node_centralities)

  return edges,nodes


# In[69]:


[edges,nodes] = calculate_positions(thresh_stack)


# In[70]:


def draw_graph(edges,nodes,title,**kwargs):

  # Draw edges
  edge_trace = go.Scatter(
      x=edges['x'], y=edges['y'],
      line=dict(width=0.5, color='#888'),
      mode='lines+markers',
      hoverinfo='text')

  # Draw nodes
  node_trace = go.Scatter(
      x=nodes['x'],
      y=nodes['y'],
      # Optional: Add labels to points *without* hovering (can get a little messy)
      mode='markers+text',
      # ...or, just add markers (no text)
      #mode='markers',
      text=nodes['name'],
      hoverinfo='text')
  filename=title.lower().replace(" ","_")

  # Color the node by its number of connections
  #node_trace.marker.color = nodes['adjacencies']
  node_trace.marker.color = nodes['centralities']
  
  # Draw figure
  fig = go.Figure(data=[edge_trace,node_trace],
            layout=go.Layout(
              title=title,
              titlefont_size=16,
              showlegend=False,
              hovermode='closest',
              margin=dict(b=20,l=5,r=5,t=120),
              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
              template='plotly_white')
              )
  
  fig.update_traces(textposition='top center')
  # Show figure
  fig.show()


# In[71]:


draw_graph(edges,nodes,'Test1')


# In[ ]:




