from federated_brain_age.constants import *

def get_task_by_id(task_id, pg):
    """ Retrieve the person id from the source id.
    """
    return pg.run_sql(
        f"SELECT * FROM {TASK} WHERE task_id='{task_id}' LIMIT 1",
        fetch_one=True,
    )
