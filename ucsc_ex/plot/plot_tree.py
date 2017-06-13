import plotly.plotly as py
import plotly.graph_objs as go

import igraph

nr_vertices = 25
v_label = map(str, range(nr_vertices))
# 2 stands for children number
G = Graph.Tree(nr_vertices, 2)
lay = G.layout('rt')

position = {k: lay[k] for k in range(nr_vertices)}
Y = [lay[k][1] for k in range(nr_vertices)]
M = max(Y)

# sequence of edges
es = EdgeSeq(G)
# list of edges
E = [e.tuple for e in G.es]

L = len(position)
Xn = [position[k][0] for k in range(L)]
Yn = [2 * M - position[k][1] for k in range(L)]
Xe = []
Ye = []
for edge in E:
    Xe+=[position[edge[0]][0],position[edge[1]][0], None]
    Ye+=[2 * M - position[edge[0]][1], 2 * M - position[edge[1]][1], None]

labels = v_label