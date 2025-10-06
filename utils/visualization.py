import matplotlib.pyplot as plt
import plotly.graph_objs as go
import numpy as np
import os
import matplotlib.cm as cm
import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict
import re

    
# set colors
import plotly.express as px
alphabets = list("ABCDEFGHIJKLMNO") 
colors = ['red', 'blue', 'green', 'cyan', 'magenta',
          'orange', 'dimgrey', 'darkviolet', 'teal', 'sienna',
          'deeppink', 'lime', 'royalblue', 'tan', 'darkkhaki',
          'gold', 'tomato', 'dodgerblue', 'mediumseagreen', 'orchid',
          'slateblue', 'crimson', 'peru', 'darkturquoise', 'chocolate',
          'firebrick', 'steelblue', 'olive', 'indigo', 'mediumvioletred']
assert len(alphabets) <= len(colors)

chain2color = dict(zip(alphabets, colors[:len(alphabets)]))

def get_com_colors(max_communities=60):
    colormap = plt.get_cmap('tab20')
    com_colors = []
    for i in range(max_communities):
        color_idx = i % 20 / 20
        # comm_id = "C" + str(i+1)
        color = f"rgba({int(colormap(color_idx)[0] * 255)}, {int(colormap(color_idx)[1] * 255)}, {int(colormap(color_idx)[2] * 255)}, 0.8)"
        com_colors.append(color)
    return com_colors

def plot_molecule(u_force_avg_positions, groups=None, title=""):
    
    if groups is not None:
        unique_strings = list(set(groups))  # Get unique strings
        string_to_numeric = {string: index for index, string in enumerate(unique_strings)}
        numeric_values = [string_to_numeric[string] for string in groups]
        
    Xg, Yg, Zg, color = [], [], [], []
    for k in range(u_force_avg_positions.shape[0]):
        Xg.append(u_force_avg_positions[k][0])
        Yg.append(u_force_avg_positions[k][1])
        Zg.append(u_force_avg_positions[k][2])
        if groups is not None:
            color.append(numeric_values[k])
        else:
            color.append("blue")

    # Create the edge trace, coloring by the attribute vector
    trace=go.Scatter3d(
        x=Xg,y=Yg,z=Zg,
        mode='markers',
        marker=dict(symbol='circle', color=color, colorscale='piyg', colorbar=dict(title='Groups'), size=2), # Picnic , colorbar=dict(title='PR score')
        hoverinfo='text'
    )
    axis=dict(showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title='')
    layout = go.Layout(
            title=title,
            width=600,
            height=600,
            showlegend=False,
            scene=dict(xaxis=dict(axis),yaxis=dict(axis),zaxis=dict(axis),),
        margin=dict(t=100),
        hovermode='closest')
    fig = go.Figure(data=[trace], layout=layout)
    fig.show()
    
def plot_molecules(molecules_list, title=""):
    
    colormap = cm.get_cmap('Spectral_r')
    colors = colormap(np.linspace(0, 1, len(molecules_list)))
    traces = [] 
    for i in range(len(molecules_list)):
        c = colors[i]
        color = f"rgba({c[0]*255}, {c[1]*255}, {c[2]*255}, {c[3]*255})"
        
        u_force_avg_positions = molecules_list[i]
        Xg, Yg, Zg = [], [], []
        for k in range(u_force_avg_positions.shape[0]):
            Xg.append(u_force_avg_positions[k][0])
            Yg.append(u_force_avg_positions[k][1])
            Zg.append(u_force_avg_positions[k][2])

        # Create the edge trace, coloring by the attribute vector
        trace=go.Scatter3d(
            x=Xg,y=Yg,z=Zg,
            mode='markers',
            marker=dict(symbol='circle', color=color, colorscale='piyg', colorbar=dict(title='Groups'), size=2), # Picnic , colorbar=dict(title='PR score')
            hoverinfo='text'
        )
        traces.append(trace)
        
    axis=dict(showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title='')
    layout = go.Layout(
            title=title,
            width=600,
            height=600,
            showlegend=False,
            scene=dict(xaxis=dict(axis),yaxis=dict(axis),zaxis=dict(axis),),
        margin=dict(t=100),
        hovermode='closest')
    fig = go.Figure(data=traces, layout=layout)
    fig.show()


def plot_sankey(communities, n_segments, com_colors=None):
    
    if com_colors is None:
        com_colors = get_com_colors(max_communities=60)
       
    community_dict = {k:v for k, v in communities.items()}

    node_dict = {}
    # Iterate over the community_dict to build node_dict
    for community, nodes in community_dict.items():
        for node, times in nodes.items():
            for time in times:
                if node not in node_dict:
                    node_dict[node] = {}
                node_dict[node][time] = community
                
    # Step 1: Build a dictionary to map each (community, time) pair to a unique node index
    label_to_index = {}
    index_to_label = []
    node_counter = 0
    for c in sorted([int(i) for i in community_dict.keys()]):
        community = str(c)
        nodes = community_dict[community]
        for node, times in nodes.items():
            for t in times:
                label = f"C{community} @ T{t}"
                if label not in label_to_index:
                    label_to_index[label] = node_counter
                    index_to_label.append(label)
                    node_counter += 1

    # Map each label to a horizontal position based on time (T0 = 0.0, T1 = 0.33, etc.)
    time_to_x = {}
    for i in range(n_segments):
        time_to_x[i] = i * 1 / n_segments
    x_positions = []
    y_positions = []
    for label in index_to_label:
        time_part = label.split(' @ T')[1]
        c_part = int(label.split(' @ T')[0][1:]) * 0.5
        t = int(time_part)
        x_positions.append(time_to_x[t])
        y_positions.append(c_part)

    # Step 3: Format for Sankey
    source = []
    target = []
    value = []

    # Prepare a list to store rows of the table
    table_rows = []

    # Iterate through all time points
    for time_from in range(n_segments - 1):  # Assuming time steps 0, 1, 2, 3
        time_to = time_from + 1
        for node, time_communities in node_dict.items():
            from_community = time_communities.get(time_from)
            to_community = time_communities.get(time_to)
            if from_community and to_community:
                table_rows.append((time_from, time_to, from_community, to_community, 1))

    # Convert to a DataFrame
    df = pd.DataFrame(table_rows, columns=['from_time', 'to_time', 'from_community', 'to_community', 'count_nodes'])

    # Group by and count nodes per transition
    df_counts = df.groupby(['from_time', 'to_time', 'from_community', 'to_community']).sum().reset_index()

    # Show the resulting DataFrame
    # 
    # print(df_counts)

    for i in range(df_counts.shape[0]):
        # print(f"C{from_community} @ T{from_time} -> C{to_community} @ T{to_time}: {count_nodes}")
        from_time, to_time, from_community, to_community, count_nodes = df_counts.iloc[i]
        source_label = f"C{from_community} @ T{from_time}"
        target_label = f"C{to_community} @ T{to_time}"
        source.append(label_to_index[source_label])
        target.append(label_to_index[target_label])
        value.append(count_nodes)

    # Optional: color by community
    colormap = plt.get_cmap('tab20')
    color_map = {}
    colors = []
    for label in index_to_label:
        comm_id = label.split('@')[0]
        colors.append(com_colors[int(comm_id[1:])-1])

    # Plot
    fig = go.Figure(data=[go.Sankey(
        # arrangement='fixed',
        node=dict(
            pad=5,
            thickness=10,
            line=dict(color="black", width=0.5),
            label=[i.split("@")[0] for i in index_to_label],
            color=colors,
            x=x_positions,  # set manual horizontal positions
            # y=y_positions
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        ))])

    fig.update_layout(
        title_text=f"Flow of Nodes Across Communities and Time Points",
        font_size=10,
        # width=600,   # width in pixels
        # height=800    # height in pixels
        width=120 * n_segments,   # width in pixels
        height=800    # height in pixels
    )
    fig.show()



# Helper to extract domain breakdown from string
def parse_domain_str(domain_str):
    return [(m[0].strip(), float(m[1])) for m in re.findall(r'([\w\-]+) \((\d+)%\)', domain_str)]

def plot_tree(df_sorted, domain_to_chain, com_colors=None):
    
    if com_colors is None:
        com_colors = get_com_colors(max_communities=60)
        
    # Collect links 
    labels = []
    node_colors = [] 
    label_index = {}
    
    def get_label(label):
        if label not in label_index:
            label_index[label] = len(labels)
            labels.append(label)
            if len(label) <= 3 and label[0] == "C":
                node_colors.append(com_colors[int(label[1:]) - 1])
            else:
                node_colors.append("rgba(135,206,250,0.6)")
        return label_index[label]

    # Sankey edges
    source = []
    target = []
    value = []

    for i in range(df_sorted.shape[0]):
        comm, chain_str, domain_str = df_sorted.iloc[i].name, df_sorted.iloc[i]["Chain"], df_sorted.iloc[i]["Domain"] 
        if len(chain_str) == 0:
            continue 
        comm_idx = get_label(comm)
        domain_info = parse_domain_str(domain_str)

        for domain_name, percent in domain_info:
            chain = domain_to_chain[domain_name]
            chain_idx = get_label(chain)
            if "loop" in domain_name:
                continue 
            domain_node = f"{domain_name}"  # Keep unique per chain
            domain_idx = get_label(domain_node)
            weight = percent / 100.0
            
            # Chain -> Domain (accumulate value)
            source.append(chain_idx)
            target.append(domain_idx)
            value.append(weight)  # as 1.0 * weight

            # Domain -> Community (always 1 full community distributed)
            source.append(domain_idx)
            target.append(comm_idx)
            value.append(weight)

    # Plot Sankey
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors, 
            # color="rgba(135,206,250,0.6)"  # light sky blue transparent
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color="rgba(169,169,169,0.4)",  # light gray flow
        )
    )])

    fig.update_layout(title_text="Sankey Diagram: Chain → Domain → Community", font_size=10, height=850, width=500)
    fig.show()
    
    
def compute_chain_intervals(atoms_in_chain):
    """
    Computes and returns a dictionary mapping each chain to its interval, ensuring no overlaps.

    Args:
        atoms_in_chain (dict): A dictionary where keys are chain identifiers and values are lists of atom indices.

    Returns:
        dict: A dictionary where keys are chain identifiers and values are tuples representing intervals (start, end).

    Raises:
        AssertionError: If any overlap is detected between the intervals of consecutive chains.

    This function processes each chain, calculates the maximum atom indices, and constructs intervals based on these max values.
    Intervals are returned in a dictionary if they do not overlap.
    """

    intervals = {}
    previous_max = -1  # Start from -1 so that the first interval starts from 0

    # Iterate over each chain to calculate intervals and store them in the dictionary
    for chain, atoms in atoms_in_chain.items():
        max_atom = np.max(atoms)
        current_interval = (previous_max + 1, max_atom)
        intervals[chain] = current_interval
        previous_max = max_atom  # Update the previous_max for the next interval

    # Check for overlaps using assert statements
    previous_interval = None
    for chain, interval in intervals.items():
        if previous_interval and previous_interval[1] >= interval[0]:
            raise AssertionError(
                f"Overlap detected between intervals {previous_interval} and {interval} for chain {chain}")
        previous_interval = interval

    # Return the dictionary of intervals if no overlaps are detected
    return intervals


def plot_graph_community(positions: dict,
                         communities: list,
                         processor: None,
                         community_idx: int,
                         force_idx: int = 0,
                         base_scale: float = 4.0,
                         std_scale: float = 10.0,
                         cone_scale: float = 0.5,
                         show_figure: bool = True,
                         save_name: str = None):
    
    assert force_idx in [0, 1]
    
    # 3D layout
    axis = dict(showbackground=False, showline=False, zeroline=False,
                showgrid=False, showticklabels=False, title='')
    layout = go.Layout(title='', width=1250, height=1250, showlegend=False,
                       scene=dict(xaxis=dict(axis), yaxis=dict(axis), zaxis=dict(axis)),
                       margin=dict(t=100), hovermode='closest')
    fig = go.Figure(layout=layout)
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                      scene=dict(dragmode='orbit'))
    
    # interval / chain / color
    intervals = compute_chain_intervals(processor.atoms_in_chain[force_idx])
    interval2chain = {v: k for k, v in intervals.items()}
    
    # color for reference (drawing graph backbone)
    reference_color = []
    for atom_id in processor.atom_ids[force_idx]:
        for interval, chain in interval2chain.items():
            if atom_id >= interval[0] and atom_id <= interval[1]:
                reference_color.append(chain2color[chain])
    reference_color = np.array(reference_color)

    # color for community
    nodes_in_community = communities[community_idx]['nodes']
    community_color = []
    for node in nodes_in_community:
        node = processor.atom_ids[force_idx][node]
        for interval, chain in interval2chain.items():
            if node >= interval[0] and node <= interval[1]:
                community_color.append(chain2color[chain])
    community_color = np.array(community_color)
                
    # Determine the dominant color for the community
    unique_colors, counts = np.unique(colors, return_counts=True)
    most_frequent_index = np.argmax(counts)
    dominant_color = unique_colors[most_frequent_index]

    # Plot structure
    fig.add_trace(go.Scatter3d(
        x=positions[force_idx]['mean'][:, 0],
        y=positions[force_idx]['mean'][:, 1],
        z=positions[force_idx]['mean'][:, 2],
        mode='markers',
        marker=dict(symbol='circle', size=base_scale, color=reference_color, opacity=0.3)
    ))

    # Plot community mean position and standard deviation
    # mean
    position_community = positions[force_idx]['mean'][nodes_in_community, :]
    fig.add_trace(go.Scatter3d(
        x=position_community[:, 0], y=position_community[:, 1], z=position_community[:, 2],
        mode='markers',
        marker=dict(symbol='circle',
                    size=np.array([base_scale + (s * std_scale) for s
                                   in positions[force_idx]['std'][nodes_in_community]]),
                    color=community_color, opacity=0.01)
    ))
    
    # std
    fig.add_trace(go.Scatter3d(
        x=position_community[:, 0], y=position_community[:, 1], z=position_community[:, 2],
        mode='markers',
        marker=dict(symbol='circle', size=base_scale, color=community_color, opacity=0.9)
    ))

    # Calculate the center of the community for each extension
    start_extension = force_idx
    end_extension = 1 - start_extension

    start_positions = positions[start_extension]['mean'][nodes_in_community, :]
    end_positions = positions[end_extension]['mean'][nodes_in_community, :]

    # Calculate the center points of the community
    start_center = np.mean(start_positions, axis=0)
    end_center = np.mean(end_positions, axis=0)

    # Add black circle for the start center
    fig.add_trace(go.Scatter3d(
        x=[start_center[0]], y=[start_center[1]], z=[start_center[2]],
        mode='markers',
        marker=dict(size=6, color='green', symbol='circle')
    ))

    # Add line from start_center to end_center
    fig.add_trace(go.Scatter3d(
        x=[start_center[0], end_center[0]], y=[start_center[1], end_center[1]], z=[start_center[2], end_center[2]],
        mode='lines',
        line=dict(color='green', width=4)
    ))

    # Add the arrowhead (cone shape) at the end of the arrow
    fig.add_trace(go.Cone(
        x=[end_center[0]], y=[end_center[1]], z=[end_center[2]],
        u=[end_center[0] - start_center[0]],
        v=[end_center[1] - start_center[1]],
        w=[end_center[2] - start_center[2]],
        sizemode="absolute",
        sizeref=cone_scale,  # Adjust the size of the cone tip
        showscale=False,
        anchor="tip",
        colorscale=[[0, 'green'], [1, 'green']]
    ))

    # Show the 3D plot with the overlayed arrow
    if show_figure:
        fig.show()

    # Save the figure as an HTML file if save_path is provided
    if save_name:
        os.makedirs('3d_plots/tcr', exist_ok=True)
        fig.write_html(os.path.join('3d_plots/tcr', save_name))


def plot_graph_community_allarrows(positions: dict,
                         communities: list,
                         processor: None,
                         force_idx: int = 0,
                         base_scale: float = 4.0,
                         std_scale: float = 10.0,
                         cone_scale: float = 0.5,
                         show_figure: bool = True,
                         proportion_communities: float = 0.5,
                         color_base: str = "community",
                         save_name: str = None):
    
    assert force_idx in [0, 1]
        
    # visualize global
    axis = dict(showbackground=False, showline=False, zeroline=False,
                showgrid=False, showticklabels=False, title='')
    layout = go.Layout(title='', width=1250, height=1250, showlegend=False,
                        scene=dict(xaxis=dict(axis), yaxis=dict(axis), zaxis=dict(axis)),
                        margin=dict(t=100), hovermode='closest')
    fig = go.Figure(layout=layout)
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', scene=dict(dragmode='orbit'))
    
    # back structure 
    position_community = positions[force_idx]['reference']
    fig.add_trace(go.Scatter3d(
        x=position_community[:, 0], y=position_community[:, 1], z=position_community[:, 2],
        mode='markers',
        marker=dict(symbol='circle', size=base_scale, color="gray", opacity=0.3)
    ))

    # select communities with score above percentile 
    # community_scores = {i: communities[i]['avg_edge_mask'] * len(communities[i]['nodes']) for i in range(len(communities))}
    # threshold = np.percentile(list(community_scores.values()), (1 - proportion_communities) * 100)  # Get 80th percentile score
    community_scores = {i: communities[i]['avg_edge_mask'] for i in range(len(communities))}
    threshold = (1-proportion_communities) * max(list(community_scores.values()))
    top_communities = [k for k, v in community_scores.items() if v >= threshold]
    # num_communities = int(len(communities) * proportion_communities)
    num_communities = len(top_communities)
    print("Showing", num_communities, "communities")
    
    if color_base == "chain":
         # interval / chain / color
        intervals = compute_chain_intervals(processor.atoms_in_chain[force_idx])
        interval2chain = {v: k for k, v in intervals.items()}
    elif color_base == "contribution":
        # Generate a list of colors
        colormap = cm.get_cmap('Spectral_r')
    elif color_base == "community":
        colormap = cm.get_cmap('Spectral_r')
    else:
        colormap = "gray"
    
    # Apply min-max normalization to avg_edge_masks
    avg_edge_masks = np.asarray([c['avg_edge_mask'] for c in communities])
    min_val = np.min(avg_edge_masks)
    max_val = np.max(avg_edge_masks)
    avg_edge_masks_normalized = (avg_edge_masks - min_val) / (max_val - min_val)
    # print("edge mask normalized:", avg_edge_masks_normalized)
    
    for i in range(len(top_communities)):
        
        community_idx = top_communities[i]
        # color for community
        nodes_in_community = communities[community_idx]['nodes']
        if color_base == "contribution": 
            c = colormap(avg_edge_masks_normalized[community_idx])
            community_color = f"rgba({c[0]*255}, {c[1]*255}, {c[2]*255}, {c[3]*255})"
        elif color_base == "community":
            # c = colors[i]
            communities[community_idx]['avg_node_index'] = np.mean(communities[community_idx]['nodes'])
            c = colormap(communities[community_idx]['avg_node_index'] / 1500)
            community_color = f"rgba({c[0]*255}, {c[1]*255}, {c[2]*255}, {c[3]*255})"
        elif color_base == "chain":
            community_color = []
            for node in nodes_in_community:
                # node = atom_ids
                for interval, chain in interval2chain.items():
                    if node >= interval[0] and node <= interval[1]:
                        community_color.append(chain2color[chain])
            community_color = np.array(community_color)
        else:
            community_color = "gray"
        
        # Plot community mean position and standard deviation
        position_community = positions[force_idx]['reference'][nodes_in_community, :]
        fig.add_trace(go.Scatter3d(
            x=position_community[:, 0], y=position_community[:, 1], z=position_community[:, 2],
            mode='markers',
            marker=dict(symbol='circle', size=base_scale, color=community_color, opacity=1.0)
        ))

        # Calculate the center of the community for each extension
        start_extension = force_idx
        end_extension = 1 - start_extension
        start_positions = positions[start_extension]['reference'][nodes_in_community, :]
        end_positions = positions[end_extension]['reference'][nodes_in_community, :]

        # Calculate the center points of the community
        start_center = np.mean(start_positions, axis=0)
        end_center = np.mean(end_positions, axis=0)

        # # Add line from start_center to end_center
        if colormap != "gray":
            fig.add_trace(go.Scatter3d(
                x=[start_center[0], end_center[0]], y=[start_center[1], end_center[1]], z=[start_center[2], end_center[2]],
                mode='lines',
                line=dict(color='green', width=4)
            ))

            # Add the arrowhead (cone shape) at the end of the arrow
            fig.add_trace(go.Cone(
                x=[end_center[0]], y=[end_center[1]], z=[end_center[2]],
                u=[end_center[0] - start_center[0]],
                v=[end_center[1] - start_center[1]],
                w=[end_center[2] - start_center[2]],
                sizemode="absolute",
                sizeref=cone_scale,  # Adjust the size of the cone tip
                showscale=False,
                anchor="tip",
                colorscale=[[0, 'green'], [1, 'green']]
            ))
            
            # Add circle for the start center
            fig.add_trace(go.Scatter3d(
                x=[start_center[0]], y=[start_center[1]], z=[start_center[2]],
                mode='markers', marker=dict(size=6, color='green', symbol='circle')
            ))
        
        else:
            # Add circle for the start center
            fig.add_trace(go.Scatter3d(
                x=[start_center[0]], y=[start_center[1]], z=[start_center[2]],
                mode='markers', marker=dict(size=6, color='red', symbol='x')
            ))

    
    # Show the 3D plot with the overlayed arrow
    if show_figure:
        fig.show()

    # Save the figure as an HTML file if save_path is provided
    if save_name:
        fig.write_html(save_name)
       

def plot_attention_entropy(node_attention, edge_attention, title):
    # entropy_values_node = -node_attention * np.log2(node_attention + 1e-10)
    # entropy_values_edge = -edge_attention * np.log2(edge_attention + 1e-10)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4), dpi=150)
    ax[0].hist(node_attention, bins=30)
    ax[0].set_xlabel('Attention Weights')
    ax[0].set_ylabel('# nodes')
    ax[1].hist(edge_attention, bins=30)
    ax[1].set_xlabel('Attention Weights')
    ax[1].set_ylabel('# edges')
    fig.suptitle(title)
    plt.show()


def plot_node_attention(Xn,Yn,Zn,Xe=[],Ye=[],Ze=[],node_attention=None, edge_attention_triple=None, colormap="Reds", title="", save_name=None):

  traces = []
  if len(Xe) > 0:
    trace1=go.Scatter3d(x=Xe,
                y=Ye,
                z=Ze,
                mode='lines',
                line=dict(width=0.2, color="gray"),
                text=edge_attention_triple,
                hoverinfo='text'
                )
    traces.append(trace1)
    
  trace2=go.Scatter3d(x=Xn,
                y=Yn,
                z=Zn,
                mode='markers',
                marker=dict(symbol='circle', size=3, color=node_attention, colorscale=colormap,colorbar=dict(title='Node Attention'), opacity=0.8), # Picnic
                text=['Node {}'.format(i) for i in range(len(Xn))],  # Text for hover
                hoverinfo='text'
                )
  traces.append(trace2)
  
  axis=dict(showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title=''
            )

  layout = go.Layout(
          title=title,
          width=800,
          height=800,
          showlegend=False,
          scene=dict(
              xaxis=dict(axis),
              yaxis=dict(axis),
              zaxis=dict(axis),
          ),
      margin=dict(
          t=100
      ),
      hovermode='closest')
  
  fig=go.Figure(data=traces, layout=layout)
  
  if save_name:
    fig.write_html(save_name)
      
  # fig.show()
  

def plot_edge_attention(Xn,Yn,Zn,Xe,Ye,Ze, edge_attention_triple, colormap="Reds", title="", save_name=None):

  trace1=go.Scatter3d(x=Xe,
               y=Ye,
               z=Ze,
               mode='lines',
               line=dict(width=3, color=edge_attention_triple, colorscale=colormap, colorbar=dict(title='Edge Attention')),
               text=edge_attention_triple,
               hoverinfo='text'
               )

  trace2=go.Scatter3d(x=Xn,
                y=Yn,
                z=Zn,
                mode='markers',
                marker=dict(symbol='circle', size=3, color="gray", opacity=0.5),
                text=['Node {}'.format(i) for i in range(len(Xn))],  # Text for hover
                hoverinfo='text'
                )

  axis=dict(showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title=''
            )

  layout = go.Layout(
          title=title,
          width=800,
          height=800,
          showlegend=False,
          scene=dict(
              xaxis=dict(axis),
              yaxis=dict(axis),
              zaxis=dict(axis),
          ),
      margin=dict(
          t=100
      ),
      hovermode='closest')
  fig=go.Figure(data=[trace1, trace2], layout=layout)
  if save_name:
    fig.write_html(save_name)
  
  # fig.show()
  
def process_attention_data(attention_data):
  
  node_attention, edge_attention, edge_attention_idx = [], [], []
  num_nodes = attention_data.shape[0]

  # split node and edge attention 
  for n1 in range(num_nodes):
    node_attention.append(attention_data[n1, n1])
    for n2 in range(n1):
      if attention_data[n2,n1] > 0:
        edge_attention.append(attention_data[n2,n1])
        edge_attention_idx.append([n2,n1])

  return node_attention, edge_attention, edge_attention_idx


def prepare_attention_for_plotting(node_attention, edge_attention, edge_attention_idx, position, normalize=False, thd_edge=95):
  
  # normalize
  edge_attention = np.array(edge_attention)
  node_attention = np.array(node_attention)
  num_nodes = node_attention.shape[0]
  if normalize:
    node_attention = (node_attention - np.min(node_attention)) / (np.max(node_attention) - np.min(node_attention))
    edge_attention = (edge_attention - np.min(edge_attention)) / (np.max(edge_attention) - np.min(edge_attention))
  
  plot_attention_entropy(node_attention, edge_attention, 'Attention Weights Distribution')
  
  # prepare data for plotting 
  Xn = [position[k][0] for k in range(num_nodes)]
  Yn = [position[k][1] for k in range(num_nodes)]
  Zn = [position[k][2] for k in range(num_nodes)]
  Xe,Ye,Ze=[],[],[]
  edge_attention_triple = []
  for edge in range(len(edge_attention)):
      if edge_attention[edge] > np.percentile(edge_attention, q=thd_edge):
          s, e = edge_attention_idx[edge][0], edge_attention_idx[edge][1]
          Xe+=[position[s][0], position[e][0], None]
          Ye+=[position[s][1], position[e][1], None]
          Ze+=[position[s][2], position[e][2], None] 
          edge_attention_triple += [edge_attention[edge], edge_attention[edge], edge_attention[edge]]
  print(f"There are {node_attention.shape[0]} nodes and {edge_attention.shape[0]} edges in total")
  print(f"Showing the top {thd_edge}% edges, which includes {len(Xe) / 3} edges.")
      
  return Xn, Yn, Zn, Xe, Ye, Ze, node_attention, edge_attention, edge_attention_triple




def generate_oriented_ellipsoid(mean, cov, n_points=30, scale=1.0):
    from numpy.linalg import eig
    """Generate a 3D ellipsoid mesh aligned with the eigenvectors of the covariance matrix."""
    # Eigen-decomposition
    eigvals, eigvecs = eig(cov)
    # Ensure real (can get complex with noisy covariances)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    
    # Radii = sqrt of eigenvalues
    rx, ry, rz = scale * np.sqrt(eigvals)
    
    # Parametric angles
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = rx * np.outer(np.cos(u), np.sin(v))
    y = ry * np.outer(np.sin(u), np.sin(v))
    z = rz * np.outer(np.ones_like(u), np.cos(v))

    # Flatten and rotate
    xyz = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=0)
    rotated_xyz = eigvecs @ xyz
    rotated_xyz = rotated_xyz.T + mean

    # Reshape for surface plot
    X = rotated_xyz[:, 0].reshape(n_points, n_points)
    Y = rotated_xyz[:, 1].reshape(n_points, n_points)
    Z = rotated_xyz[:, 2].reshape(n_points, n_points)
    return X, Y, Z


def plot_oriented_ellipsoids(molecules_list, domain_colors=None, molecule_score=None, labels=None, top_residues=None,
                              title="Oriented 3D Ellipsoids", molecules_list2=None):
    
    top_k_displacements = 5
    
    colormap = cm.get_cmap('Spectral_r')
    colors = colormap(np.linspace(0, 1, len(molecules_list)))
    traces = []
    
    if molecule_score is None:
        molecule_score = [1.0] * len(molecules_list)

    # Store centers for displacement arrow plotting
    centers1 = []
    
    for i, molecule in enumerate(molecules_list):
        
        if molecule.shape[0] == 0:
            centers1.append(None) 
            continue 
        else:
            mean = molecule.mean(axis=0)
            centers1.append(mean)
            cov = np.cov(molecule.T)
        
        X, Y, Z = generate_oriented_ellipsoid(mean, cov, n_points=30, scale=2.0)

        # Choose color by index
        # rgba_color = "darkgreen" if (i <= 12 or i >= 31) else "darkred"
        if domain_colors is None: rgba_color = "grey"
        else: rgba_color = domain_colors[i]
  
        trace = go.Surface(
            x=X, y=Y, z=Z,
            showscale=False,
            opacity=0.3,
            surfacecolor=np.ones_like(Z) * molecule_score[i] * 0.1,
            colorscale=[[0, rgba_color], [1, rgba_color]],
            hoverinfo='skip',
        )
        traces.append(trace)

        label = labels[i] if labels else f"C{i+1}"
        traces.append(go.Scatter3d(
            x=[mean[0]], y=[mean[1]], z=[mean[2]],
            mode='text',
            text=[label],
            showlegend=False,
            textposition="top center",
            textfont=dict(color="white", size=14, family="Arial"),
        ))

    # Add arrows for displacement if second list is given
    if molecules_list2 is not None:
        
        centers2 = []

        for i, molecule in enumerate(molecules_list2):
            
            if molecule.shape[0] == 0:
                centers2.append(None)
            
            mean2 = molecule.mean(axis=0)
            centers2.append(mean2)

        # Step 1: Compute displacement vectors and magnitudes
        displacements = []
        for i, (p1, p2) in enumerate(zip(centers1, centers2)):
            if p2 is None or p1 is None:
                continue 
            direction = p2 - p1
            magnitude = np.linalg.norm(direction)
            displacements.append((i, p1, p2, direction, magnitude))

        # Step 2: Sort by magnitude, descending
        displacements = sorted(displacements, key=lambda x: x[-1], reverse=True)

        # Step 3: Keep top 5
        top_displacements = displacements[:top_k_displacements]

        # Step 4: Plot arrows for top displacements only
        for i, p1, p2, direction, magnitude in top_displacements:
            if magnitude == 0:
                continue

            # Line (shaft)
            try:
                traces.append(go.Scatter3d(
                    x=[p1[0], p2[0]],
                    y=[p1[1], p2[1]],
                    z=[p1[2], p2[2]],
                    mode='lines',
                    line=dict(color='orange', width=4),
                    showlegend=False
                ))

                # Cone (arrow head)
                traces.append(go.Cone(
                    x=[p2[0]], y=[p2[1]], z=[p2[2]],
                    u=[direction[0]], v=[direction[1]], w=[direction[2]],
                    sizemode="absolute",
                    sizeref=magnitude * 0.1,
                    anchor="tip",
                    showscale=False,
                    colorscale=[[0, 'orange'], [1, 'orange']],
                    name='Top Displacement'
                ))
            except Exception as e:
                print(e)

    if top_residues is not None:
        traces.append(go.Scatter3d(
            x=list(top_residues['x']),
            y=list(top_residues['y']), 
            z=list(top_residues['z']),
            mode='markers',
            marker=dict(symbol='circle', color="blue", size=4),
        ))
        traces.append(go.Scatter3d(
            x=list(top_residues["x"]),
            y=list(top_residues["y"]),
            z=list(top_residues["z"]),
            mode='text',
            text=list(top_residues["residue"]),
            textposition='top center',
            name='Annotations'
        ))

    axis = dict(showbackground=False, showline=False, zeroline=False,
                showgrid=False, showticklabels=False, title='')

    layout = go.Layout(
        title=title,
        width=800,
        height=800,
        showlegend=False,
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis),
        margin=dict(t=50),
        hovermode='closest'
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()
