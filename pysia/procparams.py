import os

class ProcParams(object):
    """
    Parameters for the image rendering process.
    """
    
    def __init__(self, is_tesseract4):
        """
        Initializes the parameters.
        It uses different settings for Tesseract 3 and Tesseract 4.
        """

        self.language = "en"        
        self.size = 32
        self.distorsion_orientation=0
        self.is_handwritten=False
        self.width=-1
        self.alignment=1
        self.text_color="#282828"
        self.orientation=0
        self.space_width=1
        self.character_spacing=0
        self.margins=(5,5,5,5)
        self.fit=False
        self.output_mask=False
        self.fonts = []
        self.save_img = False

        if is_tesseract4:
            self.skewing_angle = 0
            self.random_skew = False
            self.blur = 1
            self.random_blur = False
            #self.background_type = 0 # 0=noisy, 1=clean, 2=texture, 3=color_texture
            self.background_type = 1 # easier settings (clear background)
            self.distorsion_type=[1, 3] #1, 3
            #self.distorsion_type=0 # very easy settings - no distortions
        else:
            self.skewing_angle = 0
            self.random_skew = False
            self.blur = 0
            self.random_blur = False
            self.background_type = 1 # 0=noisy, 1=clean, 2=texture, 3=color_texture
            self.distorsion_type=0

    def load_fonts(self, fonts_dir):
        """
        Loads the fonts for rendering from a given directory.
        """

        self.fonts.clear()
        for subdir, dirs, files in os.walk(fonts_dir):
            for f in files:
                font_file = os.path.join(subdir, f)
                self.fonts.append(font_file)

    def print_params(self, file):
        """
        Prints the parameters (optionally to a file).
        """

        print(f"language: {self.language}", file=file)
        print(f"size: {self.size}", file=file)
        print(f"skewing_angle: {self.skewing_angle}", file=file)
        print(f"random_skew: {self.random_skew}", file=file)
        print(f"blur: {self.blur}", file=file)
        print(f"random_blur: {self.random_blur}", file=file)
        print(f"background_type: {self.background_type}", file=file)
        print(f"distorsion_type: {self.distorsion_type}", file=file)
        print(f"distorsion_orientation: {self.distorsion_orientation}", file=file)
        print(f"is_handwritten: {self.is_handwritten}", file=file)
        print(f"width: {self.width}", file=file)
        print(f"alignment: {self.alignment}", file=file)
        print(f"text_color: {self.text_color}", file=file)
        print(f"orientation: {self.orientation}", file=file)
        print(f"space_width: {self.space_width}", file=file)
        print(f"character_spacing: {self.character_spacing}", file=file)
        print(f"margins: {self.margins}", file=file)
        print(f"fit: {self.fit}", file=file)
        print(f"output_mask: {self.output_mask}", file=file)
        print(f"fonts: {self.fonts}", file=file)
        print(f"save_img: {self.save_img}", file=file)

    def write(self, filepath):
        """
        Writes the parameters to a file.
        """

        with open(filepath, "w") as f:
            self.print_params(f)
