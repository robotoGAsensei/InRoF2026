import pybullet as p

def get_joint_index(body, joint_name):
    for i in range(p.getNumJoints(body)):
        info = p.getJointInfo(body, i)
        if info[1].decode("utf-8") == joint_name:
            return i
    return None


def get_link_index(body, link_name):
    for i in range(p.getNumJoints(body)):
        info = p.getJointInfo(body, i)
        if info[12].decode("utf-8") == link_name:
            return i
    return None