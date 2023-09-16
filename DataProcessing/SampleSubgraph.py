import numpy as np
import networkx as nx
import pandas as pd

def SampleSubgraph(G, size=None, sample_ratio=None, keep_edge_ratio=1.0, seed=None):
    '''
    Randomly sample a subgraph with a certain set of feature parameters from the input graph. 
    See Inputs and Outputs for usage detail.

    Inputs:
    - G: A NetworkX type graph G. The full input graph from which we sample to obtain the subgraph.
    - size: integer. The number of nodes in the sampled subgraph. (Used when wanting to sample a subgraph with a particular number of nodes)
    - sample_ratio: float between 0 and 1. (Used when wanting to sample a subraph with a certain ratio of nodes from the input graph)
        * NOTE: only pass either "size" or "sample_ratio", DO NOT PASS BOTH. 
    - keep_edge_ratio: for each originally connected edge in the input graph of selected nodes, the ratio of edges to keep in the subgraph.
                        Defaut=1, i.e. keep all originally connected edges by default.
    - seed: the random seed. Used for reproducibility only. Pass a value if you want results to be reproducable.

    Outputs:
    - subgraph: The sampled subgraph of type NetworkX graph.
    '''

    assert not(size == None and sample_ratio == None or size != None and sample_ratio !=
               None), "must pass either one of subgraph size or sample ratio"
    if sample_ratio != None:
        assert sample_ratio > 0 and sample_ratio < 1, "sample ratio must be in (0,1)"
    if size != None:
        assert size > 0 and size < nx.number_of_nodes(
            G), "subgraph size must be in (0, # of nodes in input graph)"
    assert keep_edge_ratio > 0 and keep_edge_ratio <= 1, "keep_edge_ratio must be in (0,1]"

    if seed != None:
        np.random.seed(seed)

    all_nodes = np.array(nx.nodes(G))
    if size == None:
        size = round(nx.number_of_nodes(G)*sample_ratio)
    node_choices = np.random.choice(all_nodes, size=size, replace=False)

    SG = nx.subgraph(G, node_choices)
    # This must be consistent with the type of the input graph
    subgraph = nx.MultiDiGraph(SG)

    all_edges = np.array(list(nx.edges(subgraph)))
    num_full_edge = len(all_edges)
    num_remove_edge = round(num_full_edge*(1-keep_edge_ratio))
    remove_edge_choices = np.random.choice(
        np.arange(len(all_edges)), size=num_remove_edge, replace=False)
    removed_edges = all_edges[remove_edge_choices]

    for each_edge in removed_edges:
        subgraph.remove_edge(each_edge[0], each_edge[1])

    return subgraph


def sampleHeadNodeFromGraph(G, size=None, sample_ratio=None, seed=None, report_sampled_heads=True):
    '''
    Randomly sample a subgraph with a certain set of feature parameters from the input graph. 

    We only sample some "heads", and for each head, only keep its tails that are connected by edges with predefined relations.
    Only heads with predefined relations on outgoing edges are sampled. ("predefined relations" refers to relations in 
    `relations2humanLan.keys()`)

    Inputs:
    - G: A NetworkX type DIRECTED graph G. The full input graph from which we sample to obtain the subgraph.
    - size: integer. The number of heads in the sampled subgraph. (Used when wanting to sample a subgraph with a particular number of heads)
    - sample_ratio: float between 0 and 1. (Used when wanting to sample a subraph with a certain ratio of heads out of all heads from the input graph)
        * NOTE: only pass either "size" or "sample_ratio", DO NOT PASS BOTH. 
    - seed: the random seed. Used for reproducibility only. Pass a value if you want results to be reproducable.
    - report_sampled_heads: bool value determine whether the sampled heads will be printed

    Outputs:
    - subgraph: The sampled subgraph of type NetworkX graph.
    '''
    relations2humanLan = {
        "oEffect": "PersonY or others will",
        "oReact": "PersonY or others feel",
        "oWant": "PersonY or others want",
        "xAttr": "PersonX is seen as",
        "xEffect": "PersonX will",
        "xReact": "PersonX feels",
        "xWant": "PersonX wants",
        "xNeed": "PersonX needed",
        "xIntent": "PersonX wanted",
        "isAfter" : "happens after",
        "HasSubEvent" : "includes the event or action",
        "isBefore" : "happens before",
        "HinderedBy" : "can be hindered by",
        "xReason" : "because",
    }
    all_nodes = nx.nodes(G)
    num_heads = 0
    all_heads = []
    for each_node in all_nodes:
        if G.out_degree(each_node) > 0:
            num_heads += 1
            all_heads.append(each_node)
    if seed != None:
        np.random.seed(seed)

    assert not(size == None and sample_ratio == None or size != None and sample_ratio !=
               None), "must pass either one of subgraph size or sample ratio"
    if sample_ratio != None:
        assert sample_ratio > 0 and sample_ratio <= 1, "sample ratio must be in (0,1]"
    if size != None:
        assert size <= num_heads and size > 0, (
            "# of sampled heads must be in (0, total # of heads], the total # of heads is", num_heads)

    if size == None:
        size = round(num_heads*sample_ratio)

    np.random.shuffle(all_heads)
    # head_choices = list(np.random.choice(all_heads, size=size, replace=False))
    head_choices = []
    
    for i in range(size):
        thisHeadOK = False
        while not thisHeadOK:
            ind = np.random.randint(0,len(all_heads))
            avaiRelationForThisHead = getEdgeRelations(G,all_heads[ind],'head',False)
            
            for r in avaiRelationForThisHead:
                if r in relations2humanLan.keys():
                    thisHeadOK = True
                    break
            if thisHeadOK:
                head_choices.append(all_heads[ind])

    if report_sampled_heads:
        print("The sampled heads are: \n", head_choices)
    
    tail_choices = []
    for each_head in head_choices:
        tails = list(G.successors(each_head))
        tail_choices += tails
    
    tails_to_remove = []
    for each_tail in tail_choices:
        avaiRelationForThisTail = getEdgeRelations(G, each_tail,'tail',False)
        thisTailOK = False
        for r in avaiRelationForThisTail:
            if r in relations2humanLan.keys():
                thisTailOK = True
                break
        if not thisTailOK:
            tails_to_remove.append(each_tail)

    for each_bad_tail in tails_to_remove:
        tail_choices.remove(each_bad_tail)

    SG = nx.subgraph(G, head_choices+tail_choices)
    subgraph = nx.MultiDiGraph(SG)

    return subgraph


def getHeads(G):
    '''
    get all the head nodes in the input graph G. (node with an outward arrow and no inward arrow is considered as head)

    Input: 
    - G: directed graph G
    Output:
    - heads: A list containing all head nodes in G.
    '''
    heads = []
    for each_node in nx.nodes(G):
        if G.out_degree(each_node) > 0 and G.in_degree(each_node) == 0:
            heads.append(each_node)
    return heads


def getTails(G):
    '''
    get all the tails nodes in the input graph G. (any node with an inward arrow is considered as tail)

    Input: 
    - G: directed graph G
    Output:
    - tails: A list containing all tail nodes in G.
    '''
    tails = []
    for each_node in nx.nodes(G):
        if G.in_degree(each_node) > 0:
            tails.append(each_node)
    return tails


def getEdgeRelations(G, node, nodetype, relationRestriction=False):
    '''
    get the edge relations of a given node. Only gets the relations that are in `relations2humanLan.keys()` when relationRestriction = True.

    Inputs:
    - G: the graph we are considering.
    - node: the node of interest. Pass in the node label, which in our case should be a string.
    - nodetype: either 'head' or 'tail', showing the node type of the passed in node.
    - relationRestriction: Only gets the relations that are in `relations2humanLan.keys()` when relationRestriction = True.

    Outputs:
    - relations: A list containing all relations that the given node has on its connected edges.
    '''

    relations2humanLan = {
        "oEffect": "PersonY or others will",
        "oReact": "PersonY or others feel",
        "oWant": "PersonY or others want",
        "xAttr": "PersonX is seen as",
        "xEffect": "PersonX will",
        "xReact": "PersonX feels",
        "xWant": "PersonX wants",
        "xNeed": "PersonX needed",
        "xIntent": "PersonX wanted",
        "isAfter" : "happens after",
        "HasSubEvent" : "includes the event or action",
        "isBefore" : "happens before",
        "HinderedBy" : "can be hindered by",
        "xReason" : "because",
    }

    assert nodetype == 'head' or nodetype == 'tail', "invalid node type"

    edges = None
    if nodetype == 'head':
        edges = list(G.out_edges(node))
    elif nodetype == 'tail':
        edges = list(G.in_edges(node))

    relations = []

    for each_edge in edges:
        
        edge_label = G.get_edge_data(each_edge[0],each_edge[1])[0]
       
        relationName = edge_label['relation']

        if relationName not in relations:
            if relationRestriction:
                if relationName in relations2humanLan.keys():
                    relations.append(relationName)
            else:
                relations.append(relationName)

    assert len(list(set(relations))) == len(
        relations), "There is duplicate relation in the list"

    return relations


def genTsvDependency(G,sample_ratio):
    '''
    generate the four necessary pieces of information for writing the tsv file used for GPT2 fake node generation.
    The format in a single line of the Tsv will be:

        Event [TAB] EventType [TAB] Relations
    
    - "Event" is a sentence obtained from either `head_events` or `tail_events`, 
    - "EventType" is either 'head' or 'tail',
    - "Relations" is the deduplicated set of relations connected to this event. (Only the 9 basic relations are included)

    Inputs:
    - G: The full ATOMIC2020 graph
    - sample_ratio: float between 0 and 1. (Used when wanting to sample a subraph with a certain ratio of heads out of all heads from the input graph)
        i.e., this is the ratio between the sampled heads and all heads in the graph.

    Outputs:
    - head_events: list containing all head events
    - head_relations: list containing the connected relations for each head in head_events.
    - tail_events: list containing all tail events. (The tail events have already been processed to include a 
                    proper huaman interpretable start according to the relation type)
    - tail_relations: list containing the connected relations for each tail in tail_events.
    '''

    assert sample_ratio>0 and sample_ratio<=1, "sample ratio must be in (0,1]"

    relations2humanLan = {
        "oEffect": "PersonY or others will",
        "oReact": "PersonY or others feel",
        "oWant": "PersonY or others want",
        "xAttr": "PersonX is seen as",
        "xEffect": "PersonX will",
        "xReact": "PersonX feels",
        "xWant": "PersonX wants",
        "xNeed": "PersonX needed",
        "xIntent": "PersonX wanted",
        "isAfter" : "happens after",
        "HasSubEvent" : "includes the event or action",
        "isBefore" : "happens before",
        "HinderedBy" : "can be hindered by",
        "xReason" : "because",
    }

    sg = sampleHeadNodeFromGraph(G, sample_ratio=sample_ratio, report_sampled_heads=False)
    heads = getHeads(sg)
    tails = getTails(sg)
    head_events = heads

    head_relations = []
    for each_head in heads:
        head_relations.append(getEdgeRelations(sg, each_head,'head',True))
    
    tail_relations = []
    for each_tail in tails:
        tail_relations.append(getEdgeRelations(sg, each_tail,'tail',True))

    return_tail_relations = []
    tail_events = []
    for i in range(len(tails)):
        if tails[i].split(' ')[0].lower() == 'personx' or tails[i].split(' ')[0].lower() == 'persony':
                tails[i] = tails[i][8:]
        if tails[i].split(' ')[0].lower() == 'person x' or tails[i].split(' ')[0].lower() == 'person y':
            tails[i] = tails[i][9:]
        if tails[i].split(' ')[0].lower() == 'him' or tails[i].split(' ')[0].lower() == 'his' or \
            tails[i].split(' ')[0].lower() == 'she' or tails[i].split(' ')[0].lower() == 'her':
            tails[i] = tails[i][4:]
        if tails[i].split(' ')[0].lower() == 'he':
            tails[i] = tails[i][3:]
        if tails[i].split(' ')[0].lower() == 'they' or tails[i].split(' ')[0].lower() == 'them':
            tails[i] = tails[i][5:]
        for j in range(len(tail_relations[i])):
            r = tail_relations[i][j]
            if r=='oWant' or r=='xWant' or r=='xIntent':
                if tails[i].split(' ')[0].lower() != 'to':
                    tail_events.append(relations2humanLan[tail_relations[i][j]]+' to '+tails[i])
                    return_tail_relations.append(r)
                else:
                    tail_events.append(relations2humanLan[tail_relations[i][j]]+' '+tails[i])
                    return_tail_relations.append(r)
            else:
                tail_events.append(relations2humanLan[tail_relations[i][j]]+' '+tails[i])
                return_tail_relations.append(r)
    

    return head_events, head_relations, tail_events, return_tail_relations


def negativeSample(G, num_neg_sample=None, pos_neg_ratio=None, seed=None):
    '''
    generate negative samples in a graph

    pos_neg_ratio = num_postive_sample/num_neg_sample
    '''
    assert num_neg_sample and not pos_neg_ratio or not num_neg_sample and pos_neg_ratio

    Gcpy = nx.MultiDiGraph()
    all_relations = ["oEffect","oReact","oWant","xAttr","xEffect","xReact",
                    "xWant","xNeed","xIntent","isAfter" ,"HasSubEvent" ,"isBefore" ,"HinderedBy" ,"xReason"]
    if seed != None:
        np.random.seed(seed)
    if pos_neg_ratio:
        num_neg_sample = round(nx.number_of_edges(G)/pos_neg_ratio)
    all_nodes = nx.nodes(G)
    num_heads = 0
    all_heads = []
    for each_node in all_nodes:
        if G.out_degree(each_node) > 0:
            num_heads += 1
            all_heads.append(each_node)
    np.random.shuffle(all_heads)

    all_tails = []
    for node in all_nodes:
        if node not in all_heads:
            all_tails.append(node)
    np.random.shuffle(all_tails)

    for _ in range(num_neg_sample):
        fake_edge_added = False
        while not fake_edge_added:
            ind_h = np.random.randint(0,len(all_heads))
            ind_t = np.random.randint(0,len(all_tails))
            ind_r = np.random.randint(0,len(all_relations))
            h = all_heads[ind_h]
            t = all_tails[ind_t]
            if G.has_edge(h,t):
                continue
            else:
                Gcpy.add_edge(h,t, relation=all_relations[ind_r])
                fake_edge_added = True

    return Gcpy


def produceNegCsv(G, outputFilePath):
    '''
    '''
    all_heads = []
    heads = []
    relations = []
    tails = []
    for each_node in nx.nodes(G):
        if G.out_degree(each_node) > 0:
            all_heads.append(each_node)

    for head in all_heads:
        neighboring_tails = nx.neighbors(G,head)
        for tail in neighboring_tails:
            heads.append(head)
            tails.append(tail)
            relations.append(G.get_edge_data(head,tail)[0]['relation'])

    sample = {
        'head_event':heads,
        'relation':relations,
        'tail_event':tails, 
    }
    df = pd.DataFrame(sample)
    df.to_csv(outputFilePath, sep='\t',index=False)
