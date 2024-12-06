import numpy as np

def membership_view(theta):
    if theta>90:
        return 1
    else:
        return theta/90
    


object_pos=(1,0)
subject_vel=(1,0)
object_pos=np.array(object_pos)
subject_vel=np.array(subject_vel)
theta=np.rad2deg(np.arccos(np.dot(object_pos,subject_vel)/(np.linalg.norm(object_pos)*np.linalg.norm(subject_vel))))
print(theta)
print(membership_view(theta))
