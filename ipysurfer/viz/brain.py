import ipyvolume as ipv
import ipywidgets as widgets
import numpy as np
from pythreejs import (BlendFactors, BlendingMode, Equations, ShaderMaterial,
                       Side)

from ..utils import _calculate_lut, _mesh_edges, _smoothing_matrix
from .view import views_dict, ColorBar
from .surface import Surface


class Brain(object):
    u"""Class for visualizing a brain using ipyvolume.

    It is used for creating ipyvolume meshes of the given subject's
    cortex. The activation data can be shown on a mesh using add_data
    method. Figures, meshes, activation data and other information
    are stored as attributes of a class instance.

    Parameters
    ----------
    subject_id : str
        Subject name in Freesurfer subjects dir.
    hemi : str
        Hemisphere id (ie 'lh', 'rh', 'both', or 'split'). In the case
        of 'both', both hemispheres are shown in the same window.
        In the case of 'split' hemispheres are displayed side-by-side
        in different viewing panes.
    surf : str
        freesurfer surface mesh name (ie 'white', 'inflated', etc.).
    title : str
        Title for the window.
    cortex : str, tuple, dict, or None
        Specifies how the cortical surface is rendered. Options:

            1. The name of one of the preset cortex styles:
            ``'classic'`` (default), ``'high_contrast'``,
            ``'low_contrast'``, or ``'bone'``.
            2. A color-like argument to render the cortex as a single
            color, e.g. ``'red'`` or ``(0.1, 0.4, 1.)``. Setting
            this to ``None`` is equivalent to ``(0.5, 0.5, 0.5)``.
            3. The name of a colormap used to render binarized
            curvature values, e.g., ``Grays``.
            4. A list of colors used to render binarized curvature
            values. Only the first and last colors are used. E.g.,
            ['red', 'blue'] or [(1, 0, 0), (0, 0, 1)].
            5. A container with four entries for colormap (string
            specifying the name of a colormap), vmin (float
            specifying the minimum value for the colormap), vmax
            (float specifying the maximum value for the colormap),
            and reverse (bool specifying whether the colormap
            should be reversed. E.g., ``('Greys', -1, 2, False)``.
            6. A dict of keyword arguments that is passed on to the
            call to surface.
    alpha : float in [0, 1]
        Alpha level to control opacity of the cortical surface.
    size : float or pair of floats
        The size of the window, in pixels. can be one number to specify
        a square window, or the (width, height) of a rectangular window.
    background : matplotlib color
        Color of the background.
    foreground : matplotlib color
        Color of the foreground (will be used for colorbars and text).
        None (default) will use black or white depending on the value
        of ``background``.
    figure : list of ipyvolume.Figure | None | int
        If None (default), a new window will be created with the appropriate
        views. For single view plots, the figure can be specified as int to
        retrieve the corresponding Mayavi window.
    subjects_dir : str | None
        If not None, this directory will be used as the subjects directory
        instead of the value set using the SUBJECTS_DIR environment
        variable.
    views : list | str
        views to use.
    offset : bool
        If True, aligs origin with medial wall. Useful for viewing inflated
        surface where hemispheres typically overlap (Default: True).
    show_toolbar : bool
        If True, toolbars will be shown for each view.
    offscreen : bool
        If True, rendering will be done offscreen (not shown). Useful
        mostly for generating images or screenshots, but can be buggy.
        Use at your own risk.
    interaction : str
        Can be "trackball" (default) or "terrain", i.e. a turntable-style
        camera.
    units : str
        Can be 'm' or 'mm' (default).

    Attributes
    ----------
    geo : dict
        A dictionary of ipysurfer.Surface objects for each hemisphere.
    overlays : dict
        The overlays.
    """

    def __init__(self, subject_id, hemi, surf, title=None,
                 cortex='classic', alpha=1.0, size=800, background='black',
                 foreground=None, figure=None, subjects_dir=None,
                 views=['lateral'], offset=True, show_toolbar=False,
                 offscreen=False, interaction=None, units='mm'):
        # surf =  surface
        if cortex != 'classic':
            raise NotImplementedError('Options for parameter "cortex" ' +
                                      'is not yet supported.')

        if figure is not None:
            raise NotImplementedError('figure parameter' +
                                      'has not been implemented yet.')

        if interaction is not None:
            raise NotImplementedError('interaction parameter' +
                                      'has not been implemented yet.')

        self._foreground = foreground
        self._hemi = hemi
        self._units = units
        self._title = title
        self._subject_id = subject_id
        self._views = views
        # for now only one color bar can be added
        # since it is the same for all figures
        self._colorbar_added = False
        # array of data used by TimeViewer
        self._data = {}

        # load geometry for one or both hemispheres as necessary
        offset = None if (not offset or hemi != 'both') else 0.0

        if hemi in ('both', 'split'):
            self._hemis = ('lh', 'rh')
        elif hemi in ('lh', 'rh'):
            self._hemis = (hemi, )
        else:
            raise ValueError('hemi has to be either "lh", "rh", "split", ' +
                             'or "both"')

        if isinstance(size, int):
            fig_w = size
            fig_h = size
        else:
            fig_w, fig_h = size

        self.geo = {}
        self._figures = [[] for v in views]
        self._hemi_meshes = {}
        self._overlays = {}

        for h in self._hemis:
            # Initialize a Surface object as the geometry
            geo = Surface(subject_id, h, surf, subjects_dir, offset,
                          units=self._units)
            # Load in the geometry and curvature
            geo.load_geometry()
            geo.load_curvature()
            self.geo[h] = geo

        for ri, v in enumerate(views):
            fig = ipv.figure(width=fig_w, height=fig_h, lighting=True)
            fig.animation = 0
            self._figures[ri].append(fig)
            ipv.style.box_off()
            ipv.style.axes_off()
            ipv.style.background_color(background)
            ipv.view(views_dict[v].azim, views_dict[v].elev)

            for ci, h in enumerate(self._hemis):
                if ci == 1 and hemi == 'split':
                    # create a separate figure for right hemisphere
                    fig = ipv.figure(width=fig_w, height=fig_h, lighting=True)
                    fig.animation = 0
                    self._figures[ri].append(fig)
                    ipv.style.box_off()
                    ipv.style.axes_off()
                    ipv.style.background_color(background)
                    ipv.view(views_dict[v].azim, views_dict[v].elev)

                hemi_mesh = self._plot_hemi_mesh(self.geo[h].coords,
                                                 self.geo[h].faces,
                                                 self.geo[h].grey_curv)
                self._hemi_meshes[h + '_' + v] = hemi_mesh
                ipv.squarelim()

        self._add_title()

    def add_data(self, array, fmin=None, fmax=None, thresh=None,
                 colormap="auto", alpha=1,
                 vertices=None, smoothing_steps=None, time=None,
                 time_label="time index=%d", colorbar=True,
                 hemi=None, remove_existing=None, time_label_size=None,
                 initial_time=None, scale_factor=None, vector_alpha=None,
                 fmid=None, center=None, transparent=None, verbose=None):
        u"""Display data from a numpy array on the surface.

        This provides a similar interface to
        :meth:`surfer.Brain.add_overlay`, but it displays
        it with a single colormap. It offers more flexibility over the
        colormap, and provides a way to display four-dimensional data
        (i.e., a timecourse) or five-dimensional data (i.e., a
        vector-valued timecourse).

        .. note:: ``fmin`` sets the low end of the colormap, and is separate
                  from thresh (this is a different convention from
                  :meth:`surfer.Brain.add_overlay`).

        Parameters
        ----------
        array : numpy array, shape (n_vertices[, 3][, n_times])
            Data array. For the data to be understood as vector-valued
            (3 values per vertex corresponding to X/Y/Z surface RAS),
            then ``array`` must be have all 3 dimensions.
            If vectors with no time dimension are desired, consider using a
            singleton (e.g., ``np.newaxis``) to create a "time" dimension
            and pass ``time_label=None``.
        fmin : float
            Min value in colormap (uses real min if None).
        fmid : float
            Intermediate value in colormap (middle between fmin and fmax
            if None).
        fmax : float
            Max value in colormap (uses real max if None).
        thresh : None or float
            if not None, values below thresh will not be visible
        center : float or None
            if not None, center of a divergent colormap, changes the meaning of
            fmin, fmax and fmid.
        transparent : bool
            if True: use a linear transparency between fmin and fmid and make
            values below fmin fully transparent (symmetrically for divergent
            colormaps)
        colormap : string, list of colors, or array
            name of matplotlib colormap to use, a list of matplotlib colors,
            or a custom look up table (an n x 4 array coded with RBGA values
            between 0 and 255), the default "auto" chooses a default divergent
            colormap, if "center" is given (currently "icefire"), otherwise a
            default sequential colormap (currently "rocket").
        alpha : float in [0, 1]
            alpha level to control opacity of the overlay.
        vertices : numpy array
            vertices for which the data is defined (needed if len(data) < nvtx)
        smoothing_steps : int or None
            number of smoothing steps (smoothing is used if len(data) < nvtx)
            Default : 20
        time : numpy array
            time points in the data array (if data is 2D or 3D)
        time_label : str | callable | None
            format of the time label (a format string, a function that maps
            floating point time values to strings, or None for no label)
        colorbar : bool
            whether to add a colorbar to the figure
        hemi : str | None
            If None, it is assumed to belong to the hemisphere being
            shown. If two hemispheres are being shown, an error will
            be thrown.
        remove_existing : bool
            Remove surface added by previous "add_data" call. Useful for
            conserving memory when displaying different data in a loop.
        time_label_size : int
            Font size of the time label (default 14)
        initial_time : float | None
            Time initially shown in the plot. ``None`` to use the first time
            sample (default).
        scale_factor : float | None (default)
            The scale factor to use when displaying glyphs for vector-valued
            data.
        vector_alpha : float | None
            alpha level to control opacity of the arrows. Only used for
            vector-valued data. If None (default), ``alpha`` is used.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see surfer.verbose).

        Notes
        -----
        If the data is defined for a subset of vertices (specified
        by the "vertices" parameter), a smoothing method is used to interpolate
        the data onto the high resolution surface. If the data is defined for
        subsampled version of the surface, smoothing_steps can be set to None,
        in which case only as many smoothing steps are applied until the whole
        surface is filled with non-zeros.

        Due to a Mayavi (or VTK) alpha rendering bug, ``vector_alpha`` is
        clamped to be strictly < 1.
        """
        if len(array.shape) == 3:
            raise ValueError('Vector values in "array" are not supported.')
        if thresh is not None:
            raise NotImplementedError('"threshold" parameter is' +
                                      ' not supported yet.')
        if transparent is not None:
            raise NotImplementedError('"trasparent" is not supported yet.')
        if remove_existing is not None:
            raise NotImplementedError('"remove_existing" is not' +
                                      'supported yet.')
        if time_label_size is not None:
            raise NotImplementedError('"time_label_size" is not' +
                                      'supported yet.')
        if scale_factor is not None:
            raise NotImplementedError('"scale_factor" is not supported yet.')
        if vector_alpha is not None:
            raise NotImplementedError('"vector_alpha" is not supported yet.')
        if verbose is not None:
            raise NotImplementedError('"verbose" is not supported yet.')

        hemi = self._check_hemi(hemi)
        array = np.asarray(array)

        if initial_time is None:
            time_idx = 0
        else:
            time_idx = np.argmin(abs(time - initial_time))

        self._data['time'] = time
        self._data['initial_time'] = initial_time
        self._data['time_label'] = time_label
        self._data['time_idx'] = time_idx
        # data specific for a hemi
        self._data[hemi + '_array'] = array

        if time is not None and len(array.shape) == 2:
            # we have scalar_data with time dimension
            act_data = array[:, time_idx]
        else:
            # we have scalar data without time
            act_data = array

        if center is None:
            if fmin is None:
                fmin = array.min() if array.size > 0 else 0
            if fmax is None:
                fmax = array.max() if array.size > 0 else 1
        else:
            if fmin is None:
                fmin = 0
            if fmax is None:
                fmax = np.abs(center - array).max() if array.size > 0 else 1
        if fmid is None:
            fmid = (fmin + fmax) / 2.
        _check_limits(fmin, fmid, fmax, extra='')
        self._data['alpha'] = alpha
        self._data['colormap'] = colormap
        self._data['center'] = center
        self._data['fmin'] = fmin
        self._data['fmid'] = fmid
        self._data['fmax'] = fmax

        lut = self.update_lut()

        # Create smoothing matrix if necessary
        if len(act_data) < self.geo[hemi].x.shape[0]:
            if vertices is None:
                raise ValueError('len(data) < nvtx (%s < %s): the vertices '
                                 'parameter must not be None'
                                 % (len(act_data), self.geo[hemi].x.shape[0]))
            adj_mat = _mesh_edges(self.geo[hemi].faces)
            smooth_mat = _smoothing_matrix(vertices,
                                           adj_mat,
                                           smoothing_steps)
            act_data = smooth_mat.dot(act_data)
            self._data[hemi + '_smooth_mat'] = smooth_mat
        else:
            smooth_mat = None

        # data mapping into [0, 1] interval
        dt_max = fmax
        dt_min = fmin if center is None else -1 * fmax
        k = 1 / (dt_max - dt_min)
        b = 1 - k * dt_max
        act_data = k * act_data + b
        act_data = np.clip(act_data, 0, 1)

        act_color = lut(act_data)

        self._data['k'] = k
        self._data['b'] = b

        for ri, v in enumerate(self._views):
            if self._hemi != 'split':
                ci = 0
            else:
                ci = 0 if hemi == 'lh' else 1
            ipv.pylab.current.figure = self._figures[ri][ci]
            hemi_overlay = self._plot_hemi_overlay(self.geo[hemi].coords,
                                                   self.geo[hemi].faces,
                                                   act_color)
            self._overlays[hemi + '_' + v] = hemi_overlay

        if colorbar and not self._colorbar_added:
            ColorBar(self)
            self._colorbar_added = True

    def show(self):
        u"""Display widget."""
        ipv.show()

    def update_lut(self, fmin=None, fmid=None, fmax=None):
        u"""Update color map.

        Parameters
        ----------
        fmin : float | None
            Min value in colormap.
        fmid : float | None
            Intermediate value in colormap (middle between fmin and
            fmax).
        fmax : float | None
            Max value in colormap.
        """
        alpha = self._data['alpha']
        center = self._data['center']
        colormap = self._data['colormap']
        fmin = self._data['fmin'] if fmin is None else fmin
        fmid = self._data['fmid'] if fmid is None else fmid
        fmax = self._data['fmax'] if fmax is None else fmax

        lut = _calculate_lut(colormap, alpha=alpha, fmin=fmin, fmid=fmid,
                             fmax=fmax, center=center)
        self._data['lut'] = lut
        return lut

    @property
    def overlays(self):
        return self._overlays

    @property
    def data(self):
        u"""Data used by time viewer and color bar widgets."""
        return self._data

    @property
    def views(self):
        return self._views

    @property
    def hemis(self):
        return self._hemis

    def _add_title(self):
        u"""Add title to the current figure."""
        if self._title is None:
            title = self._subject_id.capitalize()
        else:
            title = self._title

        title_w = widgets.HTML('<p style="color: %s">' % self._foreground +
                               '<b>%s</b></p>' % title)
        hboxes = (widgets.HBox(f_row) for f_row in self._figures)
        ipv.gcc().children = (title_w, *hboxes)

    def _plot_hemi_mesh(self,
                        vertices,
                        faces,
                        color='grey'):
        u"""Plot triangular format Freesurfer surface of the brain hemispheres.

        Parameters
        ----------
        vertices : numpy.array
            Array of vertex (x, y, z) coordinates, of size
            number_of_vertices x 3.
        faces : numpy.array
            Array defining mesh triangles, of size number_of_faces x 3.
        color : str | numpy.array, optional
            Color for each point/vertex/symbol, can be string format,
            examples for red:’red’, ‘#f00’, ‘#ff0000’ or ‘rgb(1,0,0),
            or rgb array of shape (N, 3). Default value is 'grey'.

        Returns
        -------
        mesh_widget : ipyvolume.Mesh
            Ipyvolume object presenting the built mesh.
        """
        x = vertices[:, 0]
        y = vertices[:, 1]
        z = vertices[:, 2]

        mesh = ipv.plot_trisurf(x, y, z, triangles=faces, color=color)

        return mesh

    def _plot_hemi_overlay(self,
                           vertices,
                           faces,
                           color):
        u"""Plot overlay of the brain hemispheres with activation data.

        Parameters
        ----------
        vertices : numpy.array
            Array of vertex (x, y, z) coordinates, of size
            number_of_vertices x 3.
        faces : numpy.array
            Array defining mesh triangles, of size number_of_faces x 3.
        color : str | numpy.array, optional
            Color for each point/vertex/symbol, can be string format,
            examples for red:’red’, ‘#f00’, ‘#ff0000’ or ‘rgb(1,0,0),
            or rgb array of shape (N, 3). Default value is 'grey'.

        Returns
        -------
        mesh_overlay : ipyvolume.Mesh
            Ipyvolume object presenting the built mesh.
        """
        x = vertices[:, 0]
        y = vertices[:, 1]
        z = vertices[:, 2]

        mesh_overlay = ipv.plot_trisurf(x, y, z, triangles=faces, color=color)

        # Tranparency and alpha blending for the new material of the mesh
        mat = ShaderMaterial()
        mat.alphaTest = 0.1
        mat.blending = BlendingMode.CustomBlending
        mat.blendDst = BlendFactors.OneMinusSrcAlphaFactor
        mat.blendEquation = Equations.AddEquation
        mat.blendSrc = BlendFactors.SrcAlphaFactor
        mat.transparent = True
        mat.side = Side.DoubleSide

        mesh_overlay.material = mat

        return mesh_overlay

    def _check_hemi(self, hemi):
        u"""Check for safe single-hemi input, returns str."""
        if hemi is None:
            if self._hemi not in ['lh', 'rh']:
                raise ValueError('hemi must not be None when both '
                                 'hemispheres are displayed')
            else:
                hemi = self._hemi
        elif hemi not in ['lh', 'rh']:
            extra = ' or None' if self._hemi in ['lh', 'rh'] else ''
            raise ValueError('hemi must be either "lh" or "rh"' + extra)
        return hemi


def _check_limits(fmin, fmid, fmax, extra='f'):
    u"""Check for monotonicity."""
    if fmin >= fmid:
        raise ValueError('%smin must be < %smid, got %0.4g >= %0.4g'
                         % (extra, extra, fmin, fmid))
    if fmid >= fmax:
        raise ValueError('%smid must be < %smax, got %0.4g >= %0.4g'
                         % (extra, extra, fmid, fmax))
