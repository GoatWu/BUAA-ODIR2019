class Person:
    def __init__(self, id, age, sex, left, right, path, keywords):
        self.id = id
        self.age = age
        self.sex = sex
        self.isLeftEye = left
        self.isRightEye = right
        self.ImagePath = path
        self.keywords = keywords
        self.Disease_person = []
        self.Disease_eye = []

    def set_disease_person(self, x):
        self.Disease_person = x

    def set_disease_eye(self, x):
        self.Disease_eye = x
