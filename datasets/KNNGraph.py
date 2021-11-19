from datasets.Generator import Gen_graph



def KNNGraph(interval,data,label,task):
    a, b = 0, interval
    graph_list = []
    while b <= len(data):
        graph_list.append(data[a:b])
        a += interval
        b += interval
    graphset = Gen_graph('KNNGraph',graph_list, label,task)
    return graphset