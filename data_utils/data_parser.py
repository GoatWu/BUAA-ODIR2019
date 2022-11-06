import logging
import pandas as pd
import csv
from functools import reduce
from .parser_rule_engine import RuleEngine
from .parser_person import Person


class DataParser:
    def __init__(self, filepath, sheet):
        self.logger = logging.getLogger('Parser')
        self.filepath = filepath
        self.sheet = pd.read_excel(self.filepath, sheet_name=sheet)
        self.person = {}

    def generate_person(self):
        engin = RuleEngine()
        for i in self.sheet.index:
            id = self.sheet['ID'][i]
            age = self.sheet['Patient Age'][i]
            sex = self.sheet['Patient Sex'][i]
            left_fundus = self.sheet['Left-Fundus'][i]
            right_fundus = self.sheet['Right-Fundus'][i]
            left_keywords = self.sheet['Left-Diagnostic Keywords'][i]
            right_keywords = self.sheet['Right-Diagnostic Keywords'][i]
            N, D, G, C = self.sheet['N'][i], self.sheet['D'][i], self.sheet['G'][i], self.sheet['C'][i]
            A, H, M, Oth = self.sheet['A'][i], self.sheet['H'][i], self.sheet['M'][i], self.sheet['O'][i]
            disease_person = [N, D, G, C, A, H, M, Oth]
            # 左眼结果
            left_eye = Person(id, age, sex, 1, 0, left_fundus, left_keywords)
            left_eye.set_disease_person(disease_person)
            disease_left_eye, is_empty = engin.process_keywords(left_keywords)
            if not is_empty and not engin.is_unused_picture(left_fundus):
                left_eye.set_disease_eye(disease_left_eye)
                self.person[left_eye.ImagePath] = left_eye
            # 右眼结果
            right_eye = Person(id, age, sex, 0, 1, right_fundus, right_keywords)
            right_eye.set_disease_person(disease_person)
            disease_right_eye, is_empty = engin.process_keywords(right_keywords)
            if not is_empty  and not engin.is_unused_picture(right_fundus):
                right_eye.set_disease_eye(disease_right_eye)
                self.person[right_eye.ImagePath] = right_eye

    def check_data(self):
        unused_imgs = 0
        for i in self.sheet.index:
            id = self.sheet['ID'][i]
            left_fundus = self.sheet['Left-Fundus'][i]
            right_fundus = self.sheet['Right-Fundus'][i]
            left_eye = self.person[left_fundus] if (left_fundus in self.person) else None
            right_eye = self.person[right_fundus] if (right_fundus in self.person) else None
            if left_eye is None and right_eye is None:
                unused_imgs += 2
                self.logger.debug(
                    "Left and Right fundus not found: [" + left_fundus + "],[" +
                    right_fundus + "] as they have been discarded")
                continue
            if left_eye is None:
                unused_imgs += 1
                self.logger.debug(
                    "Left fundus not found: [" + left_fundus + "] as it have been discarded, "
                                                               "working with right fundus only.")
            if right_eye is None:
                unused_imgs += 1
                self.logger.debug(
                    "Right fundus not found: [" + right_fundus + "] as it have been discarded, "
                                                                 "working with left fundus only.")
            empty_vector = [1, 0, 0, 0, 0, 0, 0, 0]
            left_eye_disease = left_eye.Disease_eye if left_eye is not None else empty_vector
            right_eye_disease = right_eye.Disease_eye if right_eye is not None else empty_vector
            N = left_eye_disease[0] and right_eye_disease[0]
            D = left_eye_disease[1] or right_eye_disease[1]
            G = left_eye_disease[2] or right_eye_disease[2]
            C = left_eye_disease[3] or right_eye_disease[3]
            A = left_eye_disease[4] or right_eye_disease[4]
            H = left_eye_disease[5] or right_eye_disease[5]
            M = left_eye_disease[6] or right_eye_disease[6]
            Oth = left_eye_disease[7] or right_eye_disease[7]
            two_eye_disease = [N, D, G, C, A, H, M, Oth]
            person_disease = left_eye.Disease_person if left_eye is not None else right_eye.Disease_person
            diff, pos = self.disease_differences(person_disease, two_eye_disease)
            if diff:
                self.logger.debug("  Difference Id:" + str(id) + ", Index: " + str(pos))
                self.logger.debug("                  [N,D,G,C,A,H,M,O]")
                self.print_vector("  from source", person_disease)
                self.print_vector("  from left  ", left_eye_disease)
                self.print_vector("  from right ", right_eye_disease)
        self.logger.debug('Total discarded images: ' + str(unused_imgs))
        self.logger.debug('Total using images: ' + str(len(self.person)))

    def print_vector(self, title, vector):
        self.logger.debug(
            reduce(lambda x, y: str(x) + "{},".format(y), vector, '{}:    ['.format(title)) + ']'
        )

    @staticmethod
    def disease_differences(left_disease, right_disease):
        match = True
        position = 0
        for i in range(len(left_disease)):
            match = match and left_disease[i] == right_disease[i]
            if not match and position == 0:
                position = i
                break
        return not match, position

    def generate_csv(self, path):
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['id', 'N', 'D', 'G', 'C', 'A', 'H', 'M', 'O'])
            for i in self.sheet.index:
                left_fundus = self.sheet['Left-Fundus'][i]
                right_fundus = self.sheet['Right-Fundus'][i]
                if left_fundus in self.person:
                    left_eye = self.person[left_fundus].Disease_eye
                    N, D, G, C, A, H, M, Oth = left_eye[:]
                    writer.writerow([left_fundus, N, D, G, C, A, H, M, Oth])
                if right_fundus in self.person:
                    right_eye = self.person[right_fundus].Disease_eye
                    N, D, G, C, A, H, M, Oth = right_eye[:]
                    writer.writerow([right_fundus, N, D, G, C, A, H, M, Oth])
