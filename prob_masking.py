import numpy as np

def topological_sort(graph):
	visited = [False for i in graph]
	stack = []
	q = [0]
	while q:
		curr_node = q[0]
		q.pop()
		if visited[curr_node]:
			continue
		visited[curr_node] = True
		stack.append(curr_node)
		for i in graph[curr_node]:
			q.append(i)
	stack.reverse()
	return stack

def log_prob(graph, probs):
	q = np.array([[0 for i in graph] for j in graph])
	gr_sort = topological_sort(graph)
	for i in gr_sort:
		q[i][i] = 1
		for j in gr_sort:
			for k in graph[j]:
				q[i][k] += probs[j][k] * q[i][j]
	return np.log(q)

def main():
	graph = [[1,2], [5], [3,4], [5], [6], [6], []]
	probs = np.array([[0, 0.4, 0.6, 0, 0, 0, 0], [0,0,0,0,0,1,0,0], [0,0,0,0.8,0.2,0,0], [0,0,0,0,0,1,0], [0,0,0,0,0,0,1], [0,0,0,0,0,0,1], [0,0,0,0,0,0,0]])
	print(log_prob(graph, probs))

main()