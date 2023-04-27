

class Object3d:
    def __init__(self, label):
        self.type = label.type
        self.center_x = label.box.center_x
        self.center_y = label.box.center_y
        self.center_z = label.box.center_z
        self.height = label.box.height
        self.width = label.box.width
        self.length = label.box.length
        self.heading = label.box.heading


class LabelsLoader():
    def __init__(self, labels, labels_config=None):
        self.labels_config = labels_config
        # convert_labels_into_objects
        self.labels = labels
        self.objects_3d: list = [Object3d(label) for label in labels]