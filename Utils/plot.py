"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 02.25

Description:
    Functions needed for plotting.
"""

from scipy.stats import binned_statistic_2d
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib import colors
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path
import plotly.graph_objects as go
import numpy as np
import matplotlib.cm as cm
import plotly.express as px


def configure_matplotlib():
    """Configures Matplotlib with default settings and a modern, publication-friendly font."""
    
    # # Reset the plot configurations to default
    # plt.rcdefaults()

    # # Use a more modern serif font like Times New Roman (widely used in academic papers)
    # font_path = str(Path(matplotlib.get_data_path(), "fonts/ttf/ptserif.ttf"))  # PT Serif is a nice modern font
    # font_manager.fontManager.addfont(font_path)
    # prop = font_manager.FontProperties(fname=font_path)

    # # Apply font settings for a more professional look (serif font)
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = [prop.get_name()]  # Set the serif font to our selected font
    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams.update({'mathtext.default': 'regular'})

    # Set the global color parameters for histograms
    # plt.rcParams['hist.color'] = hist_fill_color  # Set fill color for histograms


    # Set general plot parameters for better paper visualization
    plt.rcParams['axes.titlesize'] = 14  # Title font size
    plt.rcParams['axes.labelsize'] = 16  # Axis labels font size
    plt.rcParams['xtick.labelsize'] = 14  # X-axis ticks font size
    plt.rcParams['ytick.labelsize'] = 14  # Y-axis ticks font size
    plt.rcParams['lines.linewidth'] = 2  # Line width
    plt.rcParams['lines.markersize'] = 6  # Marker size
    plt.rcParams['figure.figsize'] = [6, 4]  # Default figure size (in inches)
    plt.rcParams['savefig.dpi'] = 300  # High resolution for saving
    plt.rcParams['figure.dpi'] = 100  # For inline figure display

    # Optional: Improve grid visibility (you can comment this out if not needed)
    plt.rcParams['grid.color'] = 'gray'  # Grid line color
    plt.rcParams['grid.linestyle'] = '--'  # Grid line style
    plt.rcParams['grid.linewidth'] = 0.5  # Grid line width

    # Tight layout for avoiding overlaps
    plt.rcParams['figure.autolayout'] = True

    plt.rcParams['axes.facecolor'] = 'white'  # Ensures white background


def plot_hits_3D(x, y, z, q, q_mode='categorical', primary_vertex=None, lepton_direction=None,
                 pdg=None, energy=None, ghost=False, s=1.5, plot_label=False, name_save_html=None):
    """
    Plots hits with an interactive 3D plot using Plotly.
    - 'categorical': Uses a colormap for different types of hits.
    - 'binary': Differentiates primary leptons from the rest.
    """
    fig = go.Figure()


    # CATEGORICAL MODE (0 = Ghost, 1 = EM, 2 = Hadronic)
    if q_mode == 'categorical':
        q = np.argmax(q, axis = 1)
        color_map = {0: 'gray', 1: 'blue', 2: 'red'}
        size_map = {0: s * 1.2, 1: s * 1.4, 2: s * 1.8}  # Adjust size: FOR NOW THE SAME
        colors = [color_map[val] if val in color_map else 'black' for val in q]
        sizes = [size_map[val] if val in size_map else s for val in q]

        fig.add_trace(go.Scatter3d(
            x=z, y=x, z=y,
            mode='markers',
            marker=dict(size=sizes, color=colors, opacity=0.4),
            name='Hits'
        ))

    # BINARY MODE (Primary lepton vs Rest)
    elif q_mode == 'binary':
        print(q.shape)
        mask_lepton = q[:, 0] == 1
        mask_rest = q[:, 0] == 0

        if mask_lepton.sum() > 0:
            fig.add_trace(go.Scatter3d(
                x=z[mask_lepton], y=x[mask_lepton], z=y[mask_lepton],
                mode='markers',
                marker=dict(size=(s + 0.5), color='orange', opacity=1.0),
                name='Primary Lepton'
            ))

        if mask_rest.sum() > 0:
            fig.add_trace(go.Scatter3d(
                x=z[mask_rest], y=x[mask_rest], z=y[mask_rest],
                mode='markers',
                marker=dict(size=s, color='black', opacity=0.1),
                name='Rest'
            ))


    # ENERGY MODE (Color by sum of second and third column of q)
    elif q_mode == 'energy':
        energy_vals = q[:, 1] + q[:, 2]  # Sum second and third column
        min_energy, max_energy = np.min(energy_vals), np.max(energy_vals)
        
        # Normalize between 0 and 1
        norm_energy = (energy_vals - min_energy) / (max_energy - min_energy + 1e-6)

        # Get colors from the Viridis colormap
        color_scale = px.colors.sequential.Viridis[::-1]
        colors = [color_scale[int(val * (len(color_scale) - 1))] for val in norm_energy]

        fig.add_trace(go.Scatter3d(
                x=z, y=x, z=y,
                mode='markers',
                marker=dict(
                    size=s, 
                    color=norm_energy,  # Use normalized energy values for color mapping
                    colorscale=color_scale,
                    colorbar=dict(
                        title='Energy',  # Title for the color scale
                        tickvals=[0, 1],  # Show color scale ticks at 0 and 1
                        ticktext=[f'{min_energy:.2f}', f'{max_energy:.2f}']  # Display min and max energy
                    ),
                    opacity=0.7
                ),
                name='Energy-Based Coloring'
            ))

    # PRIMARY VERTEX
    if primary_vertex is not None and primary_vertex.shape == (3,):
        fig.add_trace(go.Scatter3d(
            x=[primary_vertex[2]], y=[primary_vertex[0]], z=[primary_vertex[1]],
            mode='markers',
            marker=dict(size=8, color='green', symbol='x'),
            name='Primary Vertex'
        ))

        # LEPTON DIRECTION (if provided)
        if lepton_direction is not None and lepton_direction.shape == (3,):
            end_point = primary_vertex + 2 * lepton_direction
            fig.add_trace(go.Scatter3d(
                x=[primary_vertex[2], end_point[2]],
                y=[primary_vertex[0], end_point[0]],
                z=[primary_vertex[1], end_point[1]],
                mode='lines',
                line=dict(color='green', width=4),
                name='Lepton Direction'
            ))

    # PLOT LABELS (PDG & Energy)
    if plot_label and pdg is not None and energy is not None:
        textstr_abs = f'Flavour: {pdg}\nEnergy: {energy:.2f} GeV'
        fig.add_annotation(
            text=textstr_abs,
            xref="paper", yref="paper",
            x=0.95, y=0.05,
            showarrow=False,
            font=dict(size=14),
            bgcolor="white",
            opacity=0.8
        )

    # 3D PLOT SETTINGS
    fig.update_layout(
        scene=dict(
            xaxis_title='Z',
            yaxis_title='X',
            zaxis_title='Y',
            xaxis=dict(range=[-1528, 1523]),
            yaxis=dict(range=[-235, 235]),
            zaxis=dict(range=[-235, 235]),
            aspectratio=dict(x=100, y=20, z=20)  # KEEPING aspect ratio
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        title='3D Hit Visualization',
        scene_camera=dict(
            eye=dict(x=-2000, y=-2000, z=1200)  # KEEPING eye position
        )
    )

    if name_save_html is not None:
        fig.write_html(name_save_html)
    fig.show()



def plot_hits(x, y, z, q, q_mode='categorical', primary_vertex=None, lepton_direction=None,
              pdg=None, energy=None, ghost=False, s=0.1, plot_label=False):
    """
    Plots hits with two different modes for `q` values:
    - 'categorical': Uses a colormap for different types of hits.
    - 'binary': Differentiates primary leptons from the rest.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

    if q_mode == 'categorical':
        cmap = ListedColormap(['gray', 'blue', 'red'])
        bounds = [-0.5, 0.5, 1.5, 2.5]  # Values just below and above the integers
        norm = BoundaryNorm(bounds, cmap.N)
        
        if ghost:
            ax.scatter(z, x, y, s=s, c="black", marker='o', alpha=0.7, label='Hits', zorder=2)
        else:
            sc = ax.scatter(z, x, y, s=s, c=q, cmap=cmap, norm=norm, marker='o', alpha=0.7, zorder=2)
        
        legend_elements = [
            Patch(facecolor='gray', label='Ghost'),
            Patch(facecolor='blue', label='Electromagnetic'),
            Patch(facecolor='red', label='Hadronic')
        ]
    
    elif q_mode == 'binary':
        mask_lepton = q[:, 0] == 1
        mask_rest = q[:, 0] == 0

        if mask_lepton.sum() > 0:
            ax.scatter(z[mask_lepton], x[mask_lepton], y[mask_lepton], s=s, c="orange", marker='o', alpha=1.0, label='Primary lepton', zorder=3)
        if mask_rest.sum() > 0:
            ax.scatter(z[mask_rest], x[mask_rest], y[mask_rest], s=s, c="black", marker='o', alpha=0.1, label='Rest', zorder=2)

        legend_elements = [
            Patch(facecolor='orange', label='Primary lepton'),
            Patch(facecolor='black', label='Rest')
        ]

    if primary_vertex is not None and primary_vertex.shape == (3,):
        ax.scatter(primary_vertex[2], primary_vertex[0], primary_vertex[1], s=200, c='green', marker='x', label='Primary Vertex', zorder=3)

        # Plot lepton direction if provided
        if lepton_direction is not None and lepton_direction.shape == (3,):
            end_point = primary_vertex + 2 * lepton_direction
            ax.quiver(primary_vertex[2], primary_vertex[0], primary_vertex[1],
                      end_point[2] - primary_vertex[2],
                      end_point[0] - primary_vertex[0],
                      end_point[1] - primary_vertex[1],
                      color='green', length=100, arrow_length_ratio=0.5, linewidth=2, zorder=5) 
    if plot_label:
        if primary_vertex is not None and primary_vertex.shape == (3,):
            legend_elements.append(plt.Line2D([0], [0], marker='x', color='green', linestyle='None', markersize=15, label='Primary Vertex'))
        if lepton_direction is not None and lepton_direction.shape == (3,):
            legend_elements.append(plt.Line2D([0], [0], marker='$\u2192$', color='green', linestyle='None', markersize=15, label='Primary lepton direction'))
        ax.legend(handles=legend_elements, loc='lower right', fontsize=20)
        if pdg is not None and energy is not None:
            textstr_abs = f'Flavour: {pdg}\nEnergy: {energy:.2f} GeV'
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax.text(5000, 10000, 1000, textstr_abs, transform=ax.transAxes,
                    fontsize=20, verticalalignment='top', horizontalalignment='right', bbox=props)
    
    ax.set_xlim(-1.5280e+03, 1.5236e+03)
    ax.set_ylim(-235, 235)
    ax.set_zlim(-235, 235)
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.set_box_aspect([300, 48, 48])
    plt.show()
