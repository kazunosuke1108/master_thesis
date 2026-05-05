import numpy as np


def get_relative_distance(data_dicts, patient, staff):
    patient_xy=np.array([data_dicts[patient]["60010000"],data_dicts[patient]["60010001"]])
    staff_xy=np.array([data_dicts[staff]["60010000"],data_dicts[staff]["60010001"]])
    return np.linalg.norm(patient_xy-staff_xy)


def assign_staff_watch_features(data_dicts, json_previous_data, structure_dict):
    patients=list(data_dicts.keys())
    staff=[patient for patient in patients if data_dicts[patient].get("50000000")=="no"]
    if len(staff)>0:
        for patient in patients:
            distances=[get_relative_distance(data_dicts,patient,staff_id) for staff_id in staff]
            closest_staff=staff[np.array(distances).argmin()]
            staff_x=data_dicts[closest_staff]["60010000"]
            staff_y=data_dicts[closest_staff]["60010001"]
            prev_x=json_previous_data.get(closest_staff+"_x",staff_x)
            prev_y=json_previous_data.get(closest_staff+"_y",staff_y)
            data_dicts[patient]["50001100"]=staff_x
            data_dicts[patient]["50001101"]=staff_y
            data_dicts[patient]["50001110"]=staff_x-prev_x
            data_dicts[patient]["50001111"]=staff_y-prev_y
    else:
        for patient in patients:
            data_dicts[patient]["50001100"]=structure_dict["staff_station"]["pos"][0]
            data_dicts[patient]["50001101"]=structure_dict["staff_station"]["pos"][1]
            data_dicts[patient]["50001110"]=structure_dict["staff_station"]["direction"][0]
            data_dicts[patient]["50001111"]=structure_dict["staff_station"]["direction"][1]
    return data_dicts
