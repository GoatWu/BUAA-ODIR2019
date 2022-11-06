class RuleEngine:
    def __init__(self):
        self.not_decisive_list = [
            "lens dust",
            "optic disk photographically invisible",
            "low image quality",
            "image offset"
        ]
        self.not_classified_list = [
            "anterior segment image",
            "no fundus image"
        ]
        self.unused_picture_list = [
            '2174_right.jpg', '2175_left.jpg', '2176_left.jpg', '2177_left.jpg',
            '2177_right.jpg', '2178_right.jpg', '2179_left.jpg', '2179_right.jpg',
            '2180_left.jpg', '2180_right.jpg', '2181_left.jpg', '2181_right.jpg',
            '2182_left.jpg', '2182_right.jpg', '2957_left.jpg', '2957_right.jpg'
        ]

    def is_not_decisive(self, word):
        return word in self.not_decisive_list

    def is_not_classified(self, word):
        return word in self.not_classified_list

    def is_unused_picture(self, img_name):
        return img_name in self.unused_picture_list

    def process_keywords(self, keywords):
        keywords = keywords.replace(',', '，')
        keywords_list = [keyword.strip() for keyword in keywords.split('，')]
        N, D, G, C, A, H, M, Oth = 0, 0, 0, 0, 0, 0, 0, 0
        not_decisive, is_empty = 0, False
        for keyword in keywords_list:
            if 'normal fundus' in keyword:
                N = 1
            elif "diabetic retinopathy" in keyword or "proliferative retinopathy" in keyword:
                D = 1
            elif "glaucoma" in keyword:
                G = 1
            elif "cataract" in keyword:
                C = 1
            elif "macular degeneration" in keyword:
                A = 1
            elif "hypertensive retinopathy" in keyword:
                H = 1
            elif "myopi" in keyword:
                M = 1
            else:
                if not self.is_not_classified(keyword):
                    Oth = 1
                if self.is_not_decisive(keyword):
                    not_decisive = 1
        # 去除掉含有"镜头污点","视盘不可见","图像质量差"和"图片偏位"这几个关键词的图像
        # 去除掉含有"外眼像"和"无眼底图像"这几个关键词的图像
        if not_decisive == 1:
            N, D, G, C, A, H, M, Oth = 0, 0, 0, 0, 0, 0, 0, 0
        if N + D + G + C + A + H + M + Oth == 0:
            is_empty = True
        return [N, D, G, C, A, H, M, Oth], is_empty
