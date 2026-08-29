import numpy as np


def get_risk_rank_by_patient(patients, risks):
    sorted_indices=np.argsort(-np.asarray(risks),kind="stable")
    return {patients[patient_idx]:rank for rank,patient_idx in enumerate(sorted_indices)}
