"""stargrid-register: check and correct the registration of a microscope image to the Visium HD
capture grid from the counts, by matching the profile of counts across the nucleus edge: fewer
molecules under nuclei than just around them on most slides (the dip), or more under the nuclei and
their ring than farther away. The match is measured by the correlation ratio of multimodal image
registration, after each count is divided by the mean count around it."""
__version__ = "0.6.0"
