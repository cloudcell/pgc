#!/usr/bin/env python3

# Code for Paper: "Polymorphic Graph Classifier"
# http://dx.doi.org/10.13140/RG.2.2.15744.55041
# Design: Alexander Bikeyev
# Date: 2025-04-20
# LICENSE: AGPL v3


"""
Brain Statistics Visualization Script

This script visualizes data from brain_stats_train_epoch_*.json files in a selected stats folder.
It ensures correct numerical ordering of the epoch files.
"""

MAX_TOP_PATHWAYS = 64

import os
import json
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import filedialog, Scale, Button, Frame, Label, HORIZONTAL
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import argparse
import imageio
import tempfile
from PIL import Image
import io

def natural_sort_key(s):
    """
    Sort strings with numbers in a natural way (e.g., epoch_1, epoch_2, ..., epoch_10)
    instead of lexicographical sorting (epoch_1, epoch_10, epoch_2, ...)
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def extract_epoch_number(filename):
    """Extract the epoch number from the filename."""
    match = re.search(r'brain_stats_train_epoch_(\d+)\.json', filename)
    if match:
        return int(match.group(1))
    return 0

def load_stats_files(stats_dir):
    """Load all brain_stats_train_epoch_*.json files from the given directory."""
    pattern = os.path.join(stats_dir, "brain_stats_train_epoch_*.json")
    files = glob.glob(pattern)
    
    # Sort files by epoch number
    files.sort(key=extract_epoch_number)
    
    if not files:
        print(f"No brain_stats_train_epoch_*.json files found in {stats_dir}")
        return []
    
    print(f"Found {len(files)} epoch files.")
    print(f"First file: {os.path.basename(files[0])}")
    print(f"Last file: {os.path.basename(files[-1])}")
    
    return files

def load_json_data(file_path):
    """Load JSON data from a file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def visualize_top_blocks(data, epoch, ax=None):
    """Visualize the top blocks as a 3D scatter plot."""
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax.clear()
    
    # Extract coordinates and counts
    coords = np.array([block['coords'] for block in data['top_blocks']])
    counts = np.array([block['count'] for block in data['top_blocks']])
    
    # Normalize counts for size
    sizes = 50 * (counts / counts.max())
    
    # Create a custom colormap
    cmap = plt.cm.viridis
    
    # Plot the scatter points
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        s=sizes, c=counts, cmap=cmap, alpha=0.7
    )
    
    # Add a colorbar
    plt.colorbar(scatter, ax=ax, label='Count')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Top Blocks - Epoch {epoch}')
    
    # Find the maximum coordinate values for each dimension
    if len(coords) > 0:
        max_x = np.max(coords[:, 0]) + 0.5
        max_y = np.max(coords[:, 1]) + 0.5
        max_z = np.max(coords[:, 2]) + 0.5
    else:
        max_x = max_y = max_z = 4  # Default if no blocks
    
    # Set axis limits based on the maximum values found
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.set_zlim(0, max_z)
    
    return ax

def visualize_top_pathways(data, epoch, ax=None, top_n=5, show_legend=True, vary_line_thickness=False):
    """Visualize the top pathways as 3D lines."""
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax.clear()
    
    # Create a colormap
    cmap = plt.cm.plasma
    
    # Find the maximum coordinate values across all pathways
    all_coords = []
    for pathway_data in data['top_pathways']:
        pathway = np.array(pathway_data['pathway'])
        all_coords.append(pathway)
    
    if all_coords:
        all_coords = np.vstack(all_coords)
        max_x = np.max(all_coords[:, 0]) + 0.5
        max_y = np.max(all_coords[:, 1]) + 0.5
        max_z = np.max(all_coords[:, 2]) + 0.5
    else:
        max_x = max_y = max_z = 4  # Default if no pathways
    
    # Compute thickness normalization if needed
    counts = [pathway_data['count'] for pathway_data in data['top_pathways'][:top_n]]
    if vary_line_thickness and counts:
        min_thick = 1.5
        max_thick = 7
        min_count = min(counts)
        max_count = max(counts)
        if max_count > min_count:
            norm = lambda c: min_thick + (max_thick - min_thick) * (c - min_count) / (max_count - min_count)
        else:
            norm = lambda c: (min_thick + max_thick) / 2
    else:
        norm = lambda c: 2
    
    # Plot each pathway (up to top_n)
    for i, pathway_data in enumerate(data['top_pathways'][:top_n]):
        pathway = np.array(pathway_data['pathway'])
        count = pathway_data['count']
        color = cmap(i / max(1, top_n))
        linewidth = norm(count)
        ax.plot(pathway[:, 0], pathway[:, 1], pathway[:, 2], 
                marker='o', linestyle='-', linewidth=linewidth, 
                color=color, label=f'Pathway {i+1} (count: {count})')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Top Pathways - Epoch {epoch} (Top {top_n})')
    
    # Set axis limits based on the maximum values found
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.set_zlim(0, max_z)
    
    # Add legend if requested
    if show_legend:
        ax.legend()
    
    return ax



class BrainStatsVisualizer:
    STEREO_OFFSET = 5  # degrees for stereo separation

    def __init__(self, files=None, folder_name=None):
        self.files = files or []
        self.current_frame = 0
        self.is_playing = False
        self.interval = 1000  # Default interval in ms (1 fps)
        # Set last_loaded_folder if files and folder_name are provided
        if files and folder_name:
            self.last_loaded_folder = folder_name
        else:
            self.last_loaded_folder = None  # Track the last loaded folder
        self.top_n_pathways = 5  # Default value for top pathways
        self.folder_name = folder_name
        self.setup_ui()

    def on_frame_change(self, value):
        """Called when the frame slider value changes."""
        frame_idx = int(float(value))
        if frame_idx != self.current_frame:
            self.update_frame(frame_idx)

    def on_top_pathways_change(self, value):
        self.top_n_pathways = int(value)
        self.update_frame(self.current_frame)

    def update_frame(self, frame_idx):
        """Update the visualization with the given frame index."""
        if not self.files or frame_idx < 0 or frame_idx >= len(self.files):
            self.status_label.config(text="No data loaded or frame out of range.")
            return
        self.current_frame = frame_idx
        # Clear the figure completely, including colorbars
        self.fig.clear()
        # Recreate the axes with 3D projection
        self.ax1 = self.fig.add_subplot(121, projection='3d')
        self.ax2 = self.fig.add_subplot(122, projection='3d')
        # Load the data for the current frame
        file_path = self.files[frame_idx]
        data = load_json_data(file_path)
        if data:
            epoch = data.get('epoch', extract_epoch_number(file_path))
            # Update the frame slider if it's not the source of the update
            if self.frame_slider.get() != frame_idx:
                self.frame_slider.set(frame_idx)
            # Update the top pathways slider range
            num_pathways = len(data.get('top_pathways', []))
            if num_pathways > 0:
                self.top_pathways_slider.config(from_=1, to=MAX_TOP_PATHWAYS)
                # if self.top_n_pathways > num_pathways:
                #     self.top_n_pathways = num_pathways
                # self.top_pathways_slider.set(self.top_n_pathways)
            else:
                self.top_pathways_slider.config(from_=1, to=1)
                self.top_n_pathways = 1
                self.top_pathways_slider.set(1)
            # Visualize the data
            visualize_top_blocks(data, epoch, self.ax1)
            visualize_top_pathways(
                data, epoch, self.ax2, 
                top_n=self.top_n_pathways, 
                show_legend=self.show_legend.get(), 
                vary_line_thickness=self.vary_line_thickness.get())
            # Update the status label
            self.status_label.config(text=f"Showing epoch {epoch} ({frame_idx+1}/{len(self.files)})")
        # Redraw the canvas
        self.fig.tight_layout()
        self.canvas.draw()

    def on_speed_change(self, value):
        self.interval = int(1000 / int(value))  # Convert to interval in ms and ensure integer

    def _rotate_pathways_about_y(self, data, angle_deg):
        """Return a deep copy of data with all pathway coordinates rotated about the 3D Y axis by angle_deg."""
        import copy
        angle_rad = np.deg2rad(angle_deg)
        rot_matrix = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])
        data_rot = copy.deepcopy(data)
        for pathway in data_rot.get('top_pathways', []):
            arr = np.array(pathway['pathway'])
            arr_rot = arr @ rot_matrix.T
            pathway['pathway'] = arr_rot.tolist()
        return data_rot

    def open_stereo_view(self):
        # --- Stereo view logic fully inside this method ---
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import tkinter as tk

        # Get data for current frame
        if not self.files or self.current_frame < 0 or self.current_frame >= len(self.files):
            return
        file_path = self.files[self.current_frame]
        data = load_json_data(file_path)
        if not data:
            return
        epoch = data.get('epoch', extract_epoch_number(file_path))
        top_n = self.top_n_pathways
        show_legend = self.show_legend.get()
        vary_line_thickness = self.vary_line_thickness.get()

        # Use the same data for both eyes (no rotation)
        DEFAULT_STEREO_ANGLE = 10
        stereo_angle = DEFAULT_STEREO_ANGLE
        center_azim = 45  # Default azimuth for center view
        elev = 30         # Default elevation

        # Create window and layout
        stereo_win = tk.Toplevel(self.root)
        stereo_win.title("Stereo Pathway View")
        stereo_win.geometry("1500x800")
        instruction = tk.Label(
            stereo_win,
            text="To see 3D: cross your eyes or look through the images until they overlap. Use 'Swap Left/Right' for parallel/cross-eyed mode.",
            font=("Arial", 12)
        )
        instruction.pack(side=tk.TOP, pady=10)
        main_frame = tk.Frame(stereo_win)
        main_frame.pack(fill=tk.BOTH, expand=True)
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, padx=(30, 10), pady=10, fill=tk.BOTH, expand=True)
        gap_frame = tk.Frame(main_frame, width=40)
        gap_frame.pack(side=tk.LEFT, fill=tk.Y)
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, padx=(10, 30), pady=10, fill=tk.BOTH, expand=True)

        # Create figures and axes
        fig_left = plt.figure(figsize=(6, 6))
        ax_left = fig_left.add_subplot(111, projection='3d')
        fig_right = plt.figure(figsize=(6, 6))
        ax_right = fig_right.add_subplot(111, projection='3d')

        # Canvases
        left_canvas = FigureCanvasTkAgg(fig_left, master=left_frame)
        left_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        right_canvas = FigureCanvasTkAgg(fig_right, master=right_frame)
        right_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Axis sync state
        sync_axes_var = tk.BooleanVar(value=True)

        def set_axes_equal(ax, data):
            import numpy as np
            # data: dict with 'top_pathways', each is a dict with 'pathway': list of [x, y, z]
            all_points = np.concatenate([np.array(p['pathway']) for p in data['top_pathways']], axis=0)
            x_mid = (all_points[:,0].max() + all_points[:,0].min()) * 0.5
            y_mid = (all_points[:,1].max() + all_points[:,1].min()) * 0.5
            z_mid = (all_points[:,2].max() + all_points[:,2].min()) * 0.5
            max_range = np.array([
                all_points[:,0].max() - all_points[:,0].min(),
                all_points[:,1].max() - all_points[:,1].min(),
                all_points[:,2].max() - all_points[:,2].min()
            ]).max() / 2.0
            ax.set_xlim(x_mid - max_range, x_mid + max_range)
            ax.set_ylim(y_mid - max_range, y_mid + max_range)
            ax.set_zlim(z_mid - max_range, z_mid + max_range)

        def draw_stereo(angle):
            ax_left.clear()
            ax_right.clear()
            # Draw the same data for both eyes
            visualize_top_pathways(data, epoch, ax_left, top_n=top_n, show_legend=show_legend, vary_line_thickness=vary_line_thickness)
            visualize_top_pathways(data, epoch, ax_right, top_n=top_n, show_legend=show_legend, vary_line_thickness=vary_line_thickness)
            # Center and equalize axes for true symmetry
            set_axes_equal(ax_left, data)
            set_axes_equal(ax_right, data)
            # Set stereoscopic viewpoints
            ax_left.view_init(elev=elev, azim=center_azim - angle/2)
            ax_right.view_init(elev=elev, azim=center_azim + angle/2)
            if sync_axes_var.get():
                # Synchronize axis limits (optional, but should already be equal)
                xlims = ax_left.get_xlim()
                ax_right.set_xlim(xlims)
                ylims = ax_left.get_ylim()
                ax_right.set_ylim(ylims)
                zlims = ax_left.get_zlim()
                ax_right.set_zlim(zlims)
            left_canvas.draw()
            right_canvas.draw()

        # Initial draw
        draw_stereo(stereo_angle)

        # Stereo angle slider and controls
        slider_frame = tk.Frame(stereo_win)
        slider_frame.pack(side=tk.BOTTOM, pady=5)
        angle_label = tk.Label(slider_frame, text="Stereo Angle (degrees):")
        angle_label.pack(side=tk.LEFT)
        angle_slider = tk.Scale(slider_frame, from_=-30, to=30, orient=tk.HORIZONTAL, length=300)
        angle_slider.set(DEFAULT_STEREO_ANGLE)
        angle_slider.pack(side=tk.LEFT)

        # Mouse wheel support for angle slider
        def on_mousewheel(event):
            # For Windows/Mac, event.delta is a multiple of 120; for Linux, event.num is 4/5
            if hasattr(event, 'delta') and event.delta:
                delta = int(event.delta / 120)
            elif hasattr(event, 'num'):
                # Linux: 4 is up, 5 is down
                if event.num == 4:
                    delta = 1
                elif event.num == 5:
                    delta = -1
                else:
                    delta = 0
            else:
                delta = 0
            if delta != 0:
                new_val = angle_slider.get() + delta
                angle_slider.set(max(-30, min(30, new_val)))
                draw_stereo(angle_slider.get())

        # Bind mouse wheel events (cross-platform)
        angle_slider.bind("<MouseWheel>", on_mousewheel)      # Windows and Mac
        angle_slider.bind("<Button-4>", on_mousewheel)        # Linux scroll up
        angle_slider.bind("<Button-5>", on_mousewheel)        # Linux scroll down

        def on_slider(val):
            angle = float(val)
            draw_stereo(angle)
        angle_slider.config(command=on_slider)

        # Reset button
        def reset_angle():
            angle_slider.set(DEFAULT_STEREO_ANGLE)
        reset_btn = tk.Button(slider_frame, text="Reset", command=reset_angle)
        reset_btn.pack(side=tk.LEFT, padx=10)

        # Axis sync checkbox
        sync_axes_cb = tk.Checkbutton(slider_frame, text="Sync Axes", variable=sync_axes_var, command=lambda: draw_stereo(angle_slider.get()))
        sync_axes_cb.pack(side=tk.LEFT, padx=10)

        # Track which canvas is on which side
        state = {'left_canvas': left_canvas, 'right_canvas': right_canvas, 'left_frame': left_frame, 'right_frame': right_frame}

        def swap_left_right():
            # Remove both canvases
            state['left_canvas'].get_tk_widget().pack_forget()
            state['right_canvas'].get_tk_widget().pack_forget()
            # Swap frames
            state['left_canvas'], state['right_canvas'] = state['right_canvas'], state['left_canvas']
            # Re-pack in swapped frames
            state['left_canvas'].get_tk_widget().pack(fill=tk.BOTH, expand=True)
            state['right_canvas'].get_tk_widget().pack(fill=tk.BOTH, expand=True)

        swap_btn = tk.Button(stereo_win, text="Swap Left/Right", command=swap_left_right)
        swap_btn.pack(side=tk.BOTTOM, pady=10)

        # Synchronize rotation
        def sync_rotation(event):
            src_ax, dst_ax = None, None
            if event.inaxes == ax_left:
                src_ax, dst_ax = ax_left, ax_right
            elif event.inaxes == ax_right:
                src_ax, dst_ax = ax_right, ax_left
            if src_ax and dst_ax:
                dst_ax.view_init(elev=src_ax.elev, azim=src_ax.azim)
                state['left_canvas'].draw()
                state['right_canvas'].draw()

        fig_left.canvas.mpl_connect('button_release_event', sync_rotation)
        fig_right.canvas.mpl_connect('button_release_event', sync_rotation)

    def reload_current_folder(self):
        """Reload the currently loaded folder and update the UI."""
        if not self.last_loaded_folder:
            self.status_label.config(text="No folder to reload. Please load a folder first.")
            return
        files = load_stats_files(self.last_loaded_folder)
        if not files:
            self.status_label.config(text=f"No brain_stats_train_epoch_*.json files found in {self.last_loaded_folder}")
            return
        self.files = files
        self.frame_slider.config(from_=0, to=len(self.files)-1)
        self.update_frame(0)
        self.status_label.config(text=f"Reloaded folder: {self.last_loaded_folder}")

    def open_stats_folder(self):
        """Open a stats folder and load the brain stats files."""
        stats_dir = select_stats_folder()
        if not stats_dir:
            return
        # Load the files
        files = load_stats_files(stats_dir)
        if not files:
            self.status_label.config(text=f"No brain_stats_train_epoch_*.json files found in {stats_dir}")
            return
        self.files = files
        self.last_loaded_folder = stats_dir
        self.frame_slider.config(from_=0, to=len(self.files)-1)
        self.update_frame(0)
        self.status_label.config(text=f"Loaded folder: {stats_dir}")
        # Update window title with the folder name
        folder_name = os.path.basename(os.path.normpath(stats_dir))
        self.root.title(f"PGC Training Stats Visualizer - {folder_name}")

    def setup_ui(self):
        # Create the main window
        self.root = tk.Tk()
        title = "PGC Training Stats Visualizer"
        if getattr(self, 'folder_name', None):
            title += f" - {self.folder_name}"
        self.root.title(title)
        self.root.geometry("1200x800")
        # Set window icon (cross-platform)
        import sys
        try:
            if sys.platform.startswith('win'):
                self.root.iconbitmap("./assets/CLOUDCELL-32x32.ico")
            else:
                icon_img = tk.PhotoImage(file="./assets/CLOUDCELL-32x32-0.png")
                self.root.wm_iconphoto(True, icon_img)
        except Exception as e:
            print(f"Warning: Could not load application icon: {e}")
        # Set proper shutdown behavior for the close button
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create a frame for the matplotlib figure
        self.fig_frame = Frame(self.root)
        self.fig_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create the matplotlib figure
        self.fig = plt.figure(figsize=(12, 6))
        self.ax1 = self.fig.add_subplot(121, projection='3d')
        self.ax2 = self.fig.add_subplot(122, projection='3d')
        
        # Embed the matplotlib figure in the tkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.fig_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add the matplotlib toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.fig_frame)
        self.toolbar.update()
        
        # Create a frame for the controls
        self.control_frame = Frame(self.root)
        self.control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Add a slider for the current frame
        self.frame_label = Label(self.control_frame, text="Frame:")
        self.frame_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.frame_slider = Scale(self.control_frame, from_=0, to=max(len(self.files)-1, 1), 
                                 orient=HORIZONTAL, length=300, 
                                 command=self.on_frame_change)
        self.frame_slider.pack(side=tk.LEFT, padx=(0, 20))
        
        # Add a slider for the playback speed (logarithmic scale)
        self.speed_label = Label(self.control_frame, text="Speed (fps):")
        self.speed_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.speed_slider = Scale(self.control_frame, from_=1, to=100, 
                                 orient=HORIZONTAL, length=200,
                                 label="",
                                 command=self.on_speed_change)
        self.speed_slider.set(10)  # Default speed (about 1 fps)
        self.speed_slider.pack(side=tk.LEFT, padx=(0, 20))
        
        # Add current speed display label
        self.speed_display = Label(self.control_frame, text="1.0 fps")
        self.speed_display.pack(side=tk.LEFT, padx=(0, 20))
        
        # Add slider for top N pathways
        self.top_n_pathways = 5
        self.top_pathways_label = Label(self.control_frame, text="Top Pathways:")
        self.top_pathways_label.pack(side=tk.LEFT, padx=(0, 5))
        self.top_pathways_slider = Scale(self.control_frame, from_=1, to=5, orient=HORIZONTAL, length=150, command=self.on_top_pathways_change)
        self.top_pathways_slider.set(self.top_n_pathways)
        self.top_pathways_slider.pack(side=tk.LEFT, padx=(0, 20))
        
        # Add play/pause button
        self.play_button = Button(self.control_frame, text="Play", 
                                 command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Add record GIF button
        self.record_button = Button(self.control_frame, text="Record GIF", 
                                   command=self.record_gif)
        self.record_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Add reload folder button
        self.reload_button = Button(self.control_frame, text="Reload Folder", command=self.reload_current_folder)
        self.reload_button.pack(side=tk.LEFT, padx=(0, 10))

        # Add stereo view button
        self.stereo_button = Button(self.control_frame, text="Stereo View", command=self.open_stereo_view)
        self.stereo_button.pack(side=tk.LEFT, padx=(0, 10))

        # Add tickbox for legend
        self.show_legend = tk.BooleanVar(value=False)
        self.legend_checkbox = tk.Checkbutton(self.control_frame, text="Show Legend", variable=self.show_legend, command=lambda: self.update_frame(self.current_frame))
        self.legend_checkbox.pack(side=tk.LEFT, padx=(0, 10))

        # Add tickbox for line thickness
        self.vary_line_thickness = tk.BooleanVar(value=True)
        self.thickness_checkbox = tk.Checkbutton(self.control_frame, text="Vary Line Thickness", variable=self.vary_line_thickness, command=lambda: self.update_frame(self.current_frame))
        self.thickness_checkbox.pack(side=tk.LEFT, padx=(0, 10))
        
        # Add status label
        self.status_label = Label(self.root, text="")
        self.status_label.pack(side=tk.BOTTOM, pady=(0, 10))
        
        # Initialize the speed with the default value
        self.on_speed_change(self.speed_slider.get())
        
        # Draw the initial frame if files are loaded
        if self.files:
            self.update_frame(0)
        else:
            self.status_label.config(text="No data loaded. Use File > Open to load brain stats files.")


            
    def create_menu_bar(self):
        """Create the menu bar with File and Help menus."""
        menu_bar = tk.Menu(self.root)

        # File menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Open Stats Folder...", command=self.open_stats_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)

        # Help menu
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="Usage Guide", command=self.show_usage_guide)
        menu_bar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menu_bar)
    
    def show_usage_guide(self):
        """Show a usage guide dialog."""
        guide_text = """
Brain Stats Visualizer - Usage Guide

Navigation:
- Use the frame slider to navigate between epochs
- Use the play/pause button to animate through epochs
- Adjust the speed slider to control animation speed
- Use the matplotlib toolbar for zooming, panning, and saving images

Visualization:
- Left plot: Top blocks as 3D scatter plot
- Right plot: Top pathways as 3D lines
- Colors indicate frequency/count

Recording:
- Click "Record GIF" to save the animation as a GIF file
- For large datasets, the recorder will sample frames to keep file size reasonable

Tips:
- Use logarithmic speed control for fine adjustments at lower speeds
- Rotate the 3D plots using the mouse for better viewing angles
- Save individual frames using the matplotlib toolbar
        """
        
        guide_window = tk.Toplevel(self.root)
        guide_window.title("Usage Guide")
        guide_window.geometry("600x500")
        
        text_widget = tk.Text(guide_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, guide_text)
        text_widget.config(state=tk.DISABLED)  # Make read-only
        
        close_button = Button(guide_window, text="Close", command=guide_window.destroy)
        close_button.pack(pady=10)
    
    def show_about(self):
        """Show an about dialog."""
        about_text = """
Brain Stats Visualizer

A tool for visualizing brain statistics data from JSON files.

Features:
- 3D visualization of brain blocks and pathways
- Animation of epoch progression
- GIF recording capability
- Interactive controls

Created: April 2025
        """
        
        about_window = tk.Toplevel(self.root)
        about_window.title("About")
        about_window.geometry("400x300")
        
        text_widget = tk.Text(about_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, about_text)
        text_widget.config(state=tk.DISABLED)  # Make read-only
    

        """Called when the speed slider value changes."""
        # Convert the slider value (1-100) to a logarithmic scale for better control
        # This gives finer control at lower speeds and broader adjustments at higher speeds
        slider_value = int(float(value))
        
        # Apply logarithmic scaling: fps ranges from 0.1 to 30
        # log_scale goes from log(1) to log(101) which is 0 to ~4.6
        # We map this to 0.1 to 30 fps
        if slider_value == 1:  # Handle the special case for slider value 1
            fps = 0.1
        else:
            log_scale = np.log(slider_value)
            fps = 0.1 + (30.0 - 0.1) * (log_scale / np.log(100))
        
        # Set the interval in milliseconds
        self.interval = int(1000 / fps)
        
        # Update the display with the actual fps value (rounded to 1 decimal place)
        self.speed_display.config(text=f"{fps:.1f} fps")
    
    def toggle_play(self):
        """Toggle between play and pause."""
        if self.is_playing:
            # Stop playing
            self.is_playing = False
            self.play_button.config(text="Play")
            if hasattr(self, 'play_job'):
                self.root.after_cancel(self.play_job)
        else:
            # Start playing
            self.is_playing = True
            self.play_button.config(text="Pause")
            self.play_next_frame()
    
    def play_next_frame(self):
        """Play the next frame and schedule the next update."""
        if not self.is_playing:
            return
        
        # Calculate the next frame index (with wraparound)
        next_frame = (self.current_frame + 1) % len(self.files)
        
        # Update the visualization
        self.update_frame(next_frame)
        
        # Schedule the next update
        self.play_job = self.root.after(self.interval, self.play_next_frame)
    
    def record_gif(self):
        """Record the animation as a GIF."""
        # Ask for the output file
        output_file = filedialog.asksaveasfilename(
            title="Save GIF Animation",
            defaultextension=".gif",
            filetypes=[("GIF files", "*.gif")]
        )
        
        if not output_file:
            return
        
        # Disable the UI during recording
        self.status_label.config(text="Recording GIF... Please wait.")
        self.record_button.config(state=tk.DISABLED)
        self.play_button.config(state=tk.DISABLED)
        self.frame_slider.config(state=tk.DISABLED)
        self.speed_slider.config(state=tk.DISABLED)
        self.root.update()
        
        try:
            frames = []
            buffers = []  # Keep references to buffers to prevent them from being garbage collected
            
            # Determine how many frames to capture (use every Nth frame if too many)
            total_frames = len(self.files)
            max_frames = 100000  # Maximum number of frames for the GIF
            
            if total_frames > max_frames:
                frame_step = total_frames // max_frames
            else:
                frame_step = 1
            
            # Capture frames
            for i in range(0, total_frames, frame_step):
                # Update the status
                self.status_label.config(text=f"Recording frame {i+1}/{total_frames}...")
                self.root.update()
                
                # Update the visualization
                self.update_frame(i)
                
                # Save the current figure to a buffer
                buf = io.BytesIO()
                self.fig.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                
                # Open the image with PIL and append to frames
                img = Image.open(buf)
                img = img.copy()  # Create a copy that doesn't depend on the buffer
                frames.append(img)
                
                # Keep a reference to the buffer
                buffers.append(buf)
            
            # Save the frames as a GIF
            self.status_label.config(text="Saving GIF...")
            self.root.update()
            
            # Save with PIL
            if frames:
                frames[0].save(
                    output_file,
                    save_all=True,
                    append_images=frames[1:],
                    optimize=False,
                    duration=self.interval,  # Use the current playback speed
                    loop=0  # Loop forever
                )
                
            self.status_label.config(text=f"GIF saved to {output_file}")
            
        except Exception as e:
            self.status_label.config(text=f"Error creating GIF: {str(e)}")
            print(f"Error creating GIF: {e}")
        finally:
            # Re-enable the UI
            self.record_button.config(state=tk.NORMAL)
            self.play_button.config(state=tk.NORMAL)
            self.frame_slider.config(state=tk.NORMAL)
            self.speed_slider.config(state=tk.NORMAL)
    
    def run(self):
        """Run the visualizer."""
        self.root.mainloop()

    def on_close(self):
        """Handle application close event for proper shutdown."""
        self.root.quit()
        self.root.destroy()


def visualize_stats_folder(stats_dir, output_file=None, show_animation=True):
    """Visualize all brain stats files in the given directory."""
    files = load_stats_files(stats_dir)
    if not files:
        return
    
    if show_animation:
        # Use the interactive visualizer
        visualizer = BrainStatsVisualizer(files)
        visualizer.run()
    else:
        # Just show the first and last epoch
        first_data = load_json_data(files[0])
        last_data = load_json_data(files[-1])
        
        if first_data and last_data:
            fig = plt.figure(figsize=(15, 12))
            
            # First epoch blocks
            ax1 = fig.add_subplot(221, projection='3d')
            visualize_top_blocks(first_data, first_data.get('epoch', extract_epoch_number(files[0])), ax1)
            
            # Last epoch blocks
            ax2 = fig.add_subplot(222, projection='3d')
            visualize_top_blocks(last_data, last_data.get('epoch', extract_epoch_number(files[-1])), ax2)
            
            # First epoch pathways
            ax3 = fig.add_subplot(223, projection='3d')
            visualize_top_pathways(first_data, first_data.get('epoch', extract_epoch_number(files[0])), ax3)
            
            # Last epoch pathways
            ax4 = fig.add_subplot(224, projection='3d')
            visualize_top_pathways(last_data, last_data.get('epoch', extract_epoch_number(files[-1])), ax4)
            
            plt.tight_layout()
            plt.show()

def select_stats_folder():
    """Open a dialog to select the stats folder."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Start in the brain_stats directory if it exists
    initial_dir = "./brain_stats"
    if not os.path.exists(initial_dir):
        initial_dir = "./"
    
    # First select the stats directory (which contains multiple experiment folders)
    stats_dir = filedialog.askdirectory(
        title="Select Brain Stats Directory",
        initialdir=initial_dir
    )
    
    if not stats_dir:
        return None
    
    # List all subdirectories in the stats directory
    subdirs = [d for d in os.listdir(stats_dir) if os.path.isdir(os.path.join(stats_dir, d))]
    
    # If there are subdirectories that look like experiment folders (stats_*)
    experiment_dirs = [d for d in subdirs if d.startswith("stats_")]
    
    if experiment_dirs:
        # Create a simple dialog to select an experiment folder
        experiment_selector = tk.Toplevel(root)
        experiment_selector.title("Select Experiment Folder")
        experiment_selector.geometry("400x300")
        
        label = tk.Label(experiment_selector, text="Select an experiment folder:")
        label.pack(pady=10)
        
        # Create a listbox for experiment selection
        listbox = tk.Listbox(experiment_selector, width=50, height=10)
        listbox.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        # Add experiment folders to the listbox
        for exp_dir in sorted(experiment_dirs):
            listbox.insert(tk.END, exp_dir)
        
        # Variable to store the selected experiment
        selected_experiment = [None]
        
        def on_select():
            selection = listbox.curselection()
            if selection:
                selected_experiment[0] = experiment_dirs[selection[0]]
            experiment_selector.destroy()
        
        # Add a select button
        select_button = tk.Button(experiment_selector, text="Select", command=on_select)
        select_button.pack(pady=10)
        
        # Wait for the user to make a selection
        experiment_selector.wait_window()
        
        if selected_experiment[0]:
            return os.path.join(stats_dir, selected_experiment[0])
    
    # If no experiment was selected or there are no experiment folders,
    # return the stats directory itself
    return stats_dir

def main():
    parser = argparse.ArgumentParser(description='Visualize brain stats from JSON files.')
    parser.add_argument('--stats-dir', type=str, help='Directory containing brain_stats_train_epoch_*.json files')
    parser.add_argument('--output', type=str, help='Output file for animation (requires ffmpeg)')
    parser.add_argument('--no-animation', action='store_true', help='Show only first and last epoch instead of animation')
    
    args = parser.parse_args()
    
    # If stats directory is provided, load files directly
    if args.stats_dir:
        files = load_stats_files(args.stats_dir)
        if not files:
            print(f"No brain_stats_train_epoch_*.json files found in {args.stats_dir}")
            return
        
        folder_name = os.path.basename(os.path.normpath(args.stats_dir))
        if args.no_animation:
            # Just show the first and last epoch
            first_data = load_json_data(files[0])
            last_data = load_json_data(files[-1])
            
            if first_data and last_data:
                fig = plt.figure(figsize=(15, 12))
                
                # First epoch blocks
                ax1 = fig.add_subplot(221, projection='3d')
                visualize_top_blocks(first_data, first_data.get('epoch', extract_epoch_number(files[0])), ax1)
                
                # Last epoch blocks
                ax2 = fig.add_subplot(222, projection='3d')
                visualize_top_blocks(last_data, last_data.get('epoch', extract_epoch_number(files[-1])), ax2)
                
                # First epoch pathways
                ax3 = fig.add_subplot(223, projection='3d')
                visualize_top_pathways(first_data, first_data.get('epoch', extract_epoch_number(files[0])), ax3)
                
                # Last epoch pathways
                ax4 = fig.add_subplot(224, projection='3d')
                visualize_top_pathways(last_data, last_data.get('epoch', extract_epoch_number(files[-1])), ax4)
                
                plt.tight_layout()
                plt.show()
        else:
            # Use the interactive visualizer with pre-loaded files
            visualizer = BrainStatsVisualizer(files, folder_name=folder_name)
            visualizer.run()
    else:
        # Try to find the latest stats_* subfolder in ./brain_stats/
        stats_root = os.path.join(os.path.dirname(__file__), "brain_stats")
        if os.path.isdir(stats_root):
            subfolders = [d for d in os.listdir(stats_root) if d.startswith("stats_") and os.path.isdir(os.path.join(stats_root, d))]
            if subfolders:
                # Sort lexicographically (timestamp format is sortable)
                latest_subfolder = sorted(subfolders)[-1]
                stats_dir = os.path.join(stats_root, latest_subfolder)
                print(f"Auto-loading latest stats folder: {stats_dir}")
                files = load_stats_files(stats_dir)
                if files:
                    folder_name = os.path.basename(os.path.normpath(stats_dir))
                    visualizer = BrainStatsVisualizer(files, folder_name=folder_name)
                    visualizer.run()
                    return
        # If not found, fall back to interactive
        visualizer = BrainStatsVisualizer()
        visualizer.run()

if __name__ == "__main__":
    main()
