from os import path as path

from nibabel import freesurfer
import numpy as np

from .funcs import _check_units, _compute_normals, _get_subjects_dir


class Surface:
    """Container for a surface.

    Attributes
    ----------
    subject_id : string
        Name of subject.
    hemi : {'lh', 'rh'}
        Which hemisphere to load.
    surf : string
        Name of the surface to load (eg. inflated, orig ...).
    subjects_dir : str | None
        If not None, this directory will be used as the subjects directory
        instead of the value set using the SUBJECTS_DIR environment variable.
    offset : float | None
        If float, align inside edge of each hemisphere to center + offset.
        If None, do not change coordinates (default).
    units : str
        Can be 'm' or 'mm' (default).
    """

    def __init__(self,
                 subject_id,
                 hemi,
                 surf,
                 subjects_dir=None,
                 offset=None,
                 units='mm'):
        """Surface constructor.

        Parameters
        ----------
        subject_id : string
            Name of subject
        hemi : {'lh', 'rh'}
            Which hemisphere to load
        surf : string
            Name of the surface to load (eg. inflated, orig ...)
        offset : float | None
            If 0.0, the surface will be offset such that the medial
            wall is aligned with the origin. If None, no offset will
            be applied. If != 0.0, an additional offset will be used.
        """
        hemis = ('lh', 'rh')

        if hemi not in hemis:
            raise ValueError('hemi should be either "lh" or "rh",' +
                             'given value {0}'.format(hemi))

        if offset is not None and ((not isinstance(offset, float)) and
                                   (not isinstance(offset, int))):
            raise ValueError('offset should either float or int, given ' +
                             'type {0}'.format(type(offset).__name__))

        self.subject_id = subject_id
        self.hemi = hemi
        self.surf = surf
        self.offset = offset
        self.units = _check_units(units)
        self.bin_curv = None
        self.coords = None
        self.curv = None
        self.faces = None
        self.grey_curv = None
        self.nn = None

        subjects_dir = _get_subjects_dir(subjects_dir)
        self.data_path = path.join(subjects_dir, subject_id)

    def load_geometry(self):
        """Load geometry of the surface.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        surf_path = path.join(self.data_path, 'surf',
                              '%s.%s' % (self.hemi, self.surf))
        coords, faces = freesurfer.read_geometry(surf_path)
        if self.units == 'm':
            coords /= 1000.
        if self.offset is not None:
            if self.hemi == 'lh':
                coords[:, 0] -= (np.max(coords[:, 0]) + self.offset)
            else:
                coords[:, 0] -= (np.min(coords[:, 0]) + self.offset)
        nn = _compute_normals(coords, faces)

        if self.coords is None:
            self.coords = coords
            self.faces = faces
            self.nn = nn
        else:
            self.coords[:] = coords
            self.faces[:] = faces
            self.nn[:] = nn

    @property
    def x(self):
        return self.coords[:, 0]

    @property
    def y(self):
        return self.coords[:, 1]

    @property
    def z(self):
        return self.coords[:, 2]

    def load_curvature(self):
        """Load in curvature values from the ?h.curv file."""
        curv_path = path.join(self.data_path, 'surf', '%s.curv' % self.hemi)
        self.curv = freesurfer.read_morph_data(curv_path)
        self.bin_curv = np.array(self.curv > 0, np.int)
        # morphometry (curvature) normalization in order to get gray cortex
        # TODO: delete self.grey_curv after cortex parameter
        # will be fully supported
        color = (self.curv > 0).astype(float)
        color = 0.5 - (color - 0.5) / 3
        color = color[:, np.newaxis] * [1, 1, 1]
        self.grey_curv = color

    def load_label(self, name):
        """Load in a Freesurfer .label file.

        Label files are just text files indicating the vertices included
        in the label. Each Surface instance has a dictionary of labels, keyed
        by the name (which is taken from the file name if not given as an
        argument.

        """
        label = freesurfer.read_label(path.join(self.data_path,
                                                'label',
                                                '%s.%s.label' %
                                                (self.hemi, name)))
        label_array = np.zeros(len(self.x), np.int)
        label_array[label] = 1
        try:
            self.labels[name] = label_array
        except AttributeError:
            self.labels = {name: label_array}

    def apply_xfm(self, mtx):
        """Apply an affine transformation matrix to the x,y,z vectors."""
        self.coords = np.dot(np.c_[self.coords, np.ones(len(self.coords))],
                             mtx.T)[:, :3]
