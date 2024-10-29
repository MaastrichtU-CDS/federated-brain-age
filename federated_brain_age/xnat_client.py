import os
import xnat

from federated_brain_age.constants import *

def retrieve_data(path):
    """ Retrieve the data from XNAT to a local folder
    """
    with xnat.connect(os.getenv(XNAT_URL), user=os.getenv(XNAT_USER), password=os.getenv(XNAT_PASSWORD)) as session:
        project = session.projects[os.getenv(XNAT_PROJECT)]
        for subject_id in project.subjects:
            subject = project.subjects[subject_id]
            for experiment_id in subject.experiments:
                resources = subject.experiments[experiment_id].resources['resources']
                resources.download('{}/resources-{}-{}.zip'.format(path, subject_id, experiment_id))
