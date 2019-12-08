"""
A CNN Model that can detect Gender:

Dataset info:
    - Dataset can be found at https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar

    - Can be loaded into google collab using 
        !wget --no-check-certificate https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar -O /tmp/faces.zip
""" 
from ModelBuilder.ModelBuilder import GenderDetectionModel
gender_model = GenderDetectionModel()
