from graphviz import Digraph

def plot_hybrid_ensemble():
    dot = Digraph(format='png')
    dot.attr(rankdir='LR', size='12')

    # ---------------- Input ----------------
    dot.node('Input', 'Input Features', shape='ellipse', style='filled', fillcolor='lightblue')

    # ---------------- BaseNN1 ----------------
    with dot.subgraph(name='cluster_base1') as c:
        c.attr(label='BaseNN1 (Shallow)', color='orange')
        c.node_attr.update(style='filled', fillcolor='orange')
        c.node('B1_fc1', 'FC1\nReLU')
        c.node('B1_fc2', 'FC2\nReLU')
        c.node('B1_dropout', 'Dropout(0.2)')
        c.node('B1_out', 'Output Logit')

        # Connections inside BaseNN1
        c.edge('B1_fc1', 'B1_fc2')
        c.edge('B1_fc2', 'B1_dropout')
        c.edge('B1_dropout', 'B1_out')

    # ---------------- BaseNN2 ----------------
    with dot.subgraph(name='cluster_base2') as c:
        c.attr(label='BaseNN2 (Deep)', color='green')
        c.node_attr.update(style='filled', fillcolor='green')
        c.node('B2_fc1', 'FC1\nTanh')
        c.node('B2_fc2', 'FC2\nTanh')
        c.node('B2_fc3', 'FC3\nTanh')
        c.node('B2_dropout', 'Dropout(0.1)')
        c.node('B2_out', 'Output Logit')

        # Connections inside BaseNN2
        c.edge('B2_fc1', 'B2_fc2')
        c.edge('B2_fc2', 'B2_fc3')
        c.edge('B2_fc3', 'B2_dropout')
        c.edge('B2_dropout', 'B2_out')

    # ---------------- Fuzzy Voting ----------------
    dot.node('Voting', 'Fuzzy Voting Layer\n(weighted avg)', shape='box', style='filled', fillcolor='plum')

    # ---------------- Final Linear ----------------
    dot.node('Final', 'Final Linear Layer\nOutput Logit', shape='box', style='filled', fillcolor='lightgrey')

    # ---------------- Connections ----------------
    dot.edge('Input', 'B1_fc1')
    dot.edge('Input', 'B2_fc1')
    dot.edge('B1_out', 'Voting', label='sigmoid')
    dot.edge('B2_out', 'Voting', label='sigmoid')
    dot.edge('Voting', 'Final')

    return dot

if __name__ == "__main__":
    dot = plot_hybrid_ensemble()
    dot.render("hybrid_ensemble_model_graph", view=True)
