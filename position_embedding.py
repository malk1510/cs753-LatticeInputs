import numpy as np

def dfs(graph, node, ldists, curr_dep):
	curr_dep += 1
	for i in graph[node]:
		if ldists[i]<curr_dep:
			ldists[i] = curr_dep
			ldists = dfs(graph, i, ldists, curr_dep)
	return ldists

def GetEmbedding(graph):
	pos = np.array([-1 for i in range(len(graph))])
	pos[0] = 0
	pos = dfs(graph, 0, pos, 0)
	return pos

if __name__ == '__main__':
	graph = [[1,2], [5], [3,4], [5], [6], [6], []]
	print(GetEmbedding(graph))